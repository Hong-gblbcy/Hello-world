import cv2
import threading
import asyncio
import time
import numpy as np
from queue import Queue
import concurrent.futures

class RTSPStreamProcessor:
    def __init__(self, rtsp_urls, max_workers=None):
        """
        初始化RTSP流处理器
        
        Args:
            rtsp_urls: RTSP URL列表
            max_workers: 最大工作线程数，默认为None（根据CPU核心数自动设置）
        """
        self.rtsp_urls = rtsp_urls
        self.streams = {}
        self.frame_queues = {}
        self.processing_queues = {}
        self.running = False
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 为每个流创建队列
        for url in self.rtsp_urls:
            self.frame_queues[url] = Queue(maxsize=30)  # 限制队列大小防止内存溢出
            self.processing_queues[url] = Queue(maxsize=10)
    
    def start(self):
        """启动所有流的处理"""
        self.running = True
        
        # 为每个RTSP流启动一个线程进行读取
        self.reader_threads = []
        for url in self.rtsp_urls:
            thread = threading.Thread(target=self._read_stream, args=(url,))
            thread.daemon = True
            thread.start()
            self.reader_threads.append(thread)
        
        # 启动处理线程 - 使用线程池处理帧
        self.processor_threads = []
        for url in self.rtsp_urls:
            thread = threading.Thread(target=self._process_frames, args=(url,))
            thread.daemon = True
            thread.start()
            self.processor_threads.append(thread)
        
        # 启动异步结果处理
        self.result_thread = threading.Thread(target=self._run_async_event_loop)
        self.result_thread.daemon = True
        self.result_thread.start()
    
    def stop(self):
        """停止所有处理"""
        self.running = False
        
        # 等待所有线程完成
        for thread in self.reader_threads:
            thread.join(timeout=1.0)
        
        for thread in self.processor_threads:
            thread.join(timeout=1.0)
        
        self.result_thread.join(timeout=1.0)
        
        # 释放所有视频捕获器
        for stream in self.streams.values():
            if stream is not None:
                stream.release()
        
        self.thread_pool.shutdown()
    
    def _read_stream(self, url):
        """
        从RTSP流读取帧的线程函数
        
        Args:
            url: RTSP流URL
        """
        print(f"Starting stream reader for {url}")
        # OpenCV捕获器使用FFMPEG后端处理RTSP
        stream = cv2.VideoCapture(url)
        self.streams[url] = stream
        
        # 配置RTSP流的缓冲区大小
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 最小化缓冲
        
        # 如果连接失败，进行重试
        retry_count = 0
        max_retries = 5
        
        while self.running:
            if not stream.isOpened():
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Failed to open {url} after {max_retries} attempts")
                    break
                    
                print(f"Attempting to reconnect to {url}, attempt {retry_count}")
                stream = cv2.VideoCapture(url)
                time.sleep(2)  # 等待重连
                continue
            
            # 重置重试计数
            retry_count = 0
            
            # 读取帧
            ret, frame = stream.read()
            if not ret:
                print(f"Failed to read frame from {url}")
                time.sleep(0.1)  # 短暂暂停避免CPU过载
                continue
            
            # 丢弃旧帧，如果队列已满
            if self.frame_queues[url].full():
                try:
                    self.frame_queues[url].get_nowait()
                except:
                    pass
            
            # 添加到队列
            timestamp = time.time()
            self.frame_queues[url].put((frame, timestamp))
    
    def _process_frames(self, url):
        """
        处理帧的线程函数，使用线程池进行并行处理
        
        Args:
            url: RTSP流URL
        """
        while self.running:
            try:
                # 获取队列中的帧
                frame, timestamp = self.frame_queues[url].get(timeout=1.0)
                
                # 使用线程池处理帧 - 这里可以是任何计算密集型操作
                future = self.thread_pool.submit(self._frame_processing, frame, url)
                
                # 添加回调以处理结果
                future.add_done_callback(self._frame_processed_callback)
                
                self.frame_queues[url].task_done()
                
            except Exception as e:
                if self.running:  # 只有在程序仍在运行时打印错误
                    if not isinstance(e, TimeoutError):  # 忽略队列超时错误
                        print(f"Error processing frame from {url}: {str(e)}")
    
    def _frame_processing(self, frame, url):
        """
        帧处理函数 - 在这里添加实际的视频处理逻辑
        
        Args:
            frame: 要处理的视频帧
            url: 来源RTSP URL
            
        Returns:
            处理结果
        """
        # 示例：转换为灰度并进行边缘检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 示例：添加时间戳
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(edges, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 返回处理结果和源URL
        return edges, url
    
    def _frame_processed_callback(self, future):
        """
        帧处理完成的回调函数
        
        Args:
            future: 处理结果的Future对象
        """
        try:
            result, url = future.result()
            # 将结果放入处理队列，供异步函数处理
            if not self.processing_queues[url].full():
                self.processing_queues[url].put(result)
        except Exception as e:
            print(f"Error in processing callback: {str(e)}")
    
    def _run_async_event_loop(self):
        """运行异步事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_result_handler())
        finally:
            loop.close()
    
    async def _async_result_handler(self):
        """异步处理结果的协程"""
        while self.running:
            # 创建异步任务处理所有队列
            tasks = []
            for url in self.rtsp_urls:
                tasks.append(self._process_result_queue(url))
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
            # 短暂睡眠避免CPU过载
            await asyncio.sleep(0.01)
    
    async def _process_result_queue(self, url):
        """
        处理单个结果队列的协程
        
        Args:
            url: RTSP流URL
        """
        while not self.processing_queues[url].empty() and self.running:
            # 使用asyncio.to_thread可以在Python 3.9+中使用
            # 这里我们使用run_in_executor来实现类似功能
            result = self.processing_queues[url].get_nowait()
            
            # 执行IO密集型操作，如保存、传输或显示
            await asyncio.get_event_loop().run_in_executor(
                None, self._handle_processed_frame, result, url
            )
            
            self.processing_queues[url].task_done()
    
    def _handle_processed_frame(self, processed_frame, url):
        """
        处理已处理帧的函数 - 执行IO操作如保存或显示
        
        Args:
            processed_frame: 处理过的帧
            url: 来源RTSP URL
        """
        # 示例：显示结果（实际应用中可能是保存到磁盘或通过网络发送）
        # cv2.imshow(f"Stream: {url}", processed_frame)
        # cv2.waitKey(1)
        
        # 示例：保存到磁盘（取消注释以启用）
        # filename = f"output/{url.replace('/', '_').replace(':', '_')}_{time.time()}.jpg"
        # cv2.imwrite(filename, processed_frame)
        
        # 在这里添加任何其他IO密集型操作
        pass


# 示例用法
def main():
    # RTSP流URL列表
    rtsp_urls = [
        "rtsp://username:password@192.168.1.100:554/stream1",
        "rtsp://username:password@192.168.1.101:554/stream1",
        "rtsp://username:password@192.168.1.102:554/stream1",
    ]
    
    # 创建并启动处理器
    processor = RTSPStreamProcessor(rtsp_urls)
    processor.start()
    
    try:
        # 让程序持续运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # 停止所有处理
        processor.stop()


if __name__ == "__main__":
    main()