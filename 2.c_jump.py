import threading
import asyncio
import time
import numpy as np
from queue import Queue
import concurrent.futures
import av  # PyAV库，用于替代OpenCV
import logging
from PIL import Image
from urllib.parse import urlparse
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTSPStreamProcessor:
    def __init__(self, rtsp_urls, max_workers=None, frame_skip=3):
        """
        初始化RTSP流处理器
        
        Args:
            rtsp_urls: RTSP URL列表
            max_workers: 最大工作线程数，默认为None（根据CPU核心数自动设置）
            frame_skip: 跳帧数，处理1帧，跳过frame_skip帧，默认为3
        """
        self.rtsp_urls = rtsp_urls
        self.streams = {}
        self.frame_queues = {}
        self.processing_queues = {}
        self.running = False
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.frame_skip = frame_skip
        
        # 跳帧计数器
        self.frame_counters = defaultdict(int)
        
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
        
        # 关闭所有视频容器
        for stream in self.streams.values():
            if stream is not None:
                stream.close()
        
        self.thread_pool.shutdown()
    
    def _read_stream(self, url):
        """
        从RTSP流读取帧的线程函数
        
        Args:
            url: RTSP流URL
        """
        logger.info(f"Starting stream reader for {url}")
        
        # 解析URL以获取流的标识
        parsed_url = urlparse(url)
        stream_id = f"{parsed_url.netloc}{parsed_url.path}"
        
        # 连接重试逻辑
        retry_count = 0
        max_retries = 5
        
        while self.running:
            try:
                # 使用PyAV打开RTSP流
                container = av.open(url, options={'rtsp_transport': 'tcp'})
                self.streams[url] = container
                
                # 获取视频流
                stream = container.streams.video[0]
                
                # 重置重试计数
                retry_count = 0
                
                # 读取帧
                for frame in container.decode(stream):
                    if not self.running:
                        break
                    
                    # 将PyAV帧转换为NumPy数组
                    img = frame.to_ndarray(format='rgb24')
                    
                    # 丢弃旧帧，如果队列已满
                    if self.frame_queues[url].full():
                        try:
                            self.frame_queues[url].get_nowait()
                        except:
                            pass
                    
                    # 添加到队列
                    timestamp = time.time()
                    self.frame_queues[url].put((img, timestamp, frame.pts))
                
                # 如果到达这里，说明流已结束
                if self.running:
                    logger.warning(f"Stream {url} ended unexpectedly")
            
            except Exception as e:
                if not self.running:
                    break
                    
                retry_count += 1
                logger.error(f"Error reading from {url}: {str(e)}")
                
                if retry_count > max_retries:
                    logger.error(f"Failed to open {url} after {max_retries} attempts")
                    break
                    
                logger.info(f"Attempting to reconnect to {url}, attempt {retry_count}")
                time.sleep(2)  # 等待重连
    
    def _process_frames(self, url):
        """
        处理帧的线程函数，使用线程池进行并行处理
        
        Args:
            url: RTSP流URL
        """
        while self.running:
            try:
                # 获取队列中的帧
                frame, timestamp, pts = self.frame_queues[url].get(timeout=1.0)
                
                # 使用线程池处理帧 - 这里可以是任何计算密集型操作
                future = self.thread_pool.submit(self._frame_processing, frame, url, timestamp, pts)
                
                # 添加回调以处理结果
                future.add_done_callback(self._frame_processed_callback)
                
                self.frame_queues[url].task_done()
                
            except Exception as e:
                if self.running:  # 只有在程序仍在运行时打印错误
                    if not isinstance(e, TimeoutError):  # 忽略队列超时错误
                        logger.error(f"Error processing frame from {url}: {str(e)}")
    
    def _frame_processing(self, frame, url, timestamp, pts):
        """
        帧处理函数 - 在这里添加实际的视频处理逻辑
        包含跳帧逻辑
        
        Args:
            frame: 要处理的视频帧 (NumPy数组)
            url: 来源RTSP URL
            timestamp: 帧抓取时间戳
            pts: 帧的presentation timestamp
            
        Returns:
            处理结果或None (如果帧被跳过)
        """
        # 实现跳帧逻辑 - 使用流URL作为键来跟踪每个流的计数器
        self.frame_counters[url] += 1
        
        # 如果不是要处理的帧，则跳过
        if (self.frame_counters[url] - 1) % (self.frame_skip + 1) != 0:
            return None, url
        
        # 此处是帧处理的实际逻辑
        logger.debug(f"Processing frame {self.frame_counters[url]} from {url}")
        
        # 示例：使用PIL进行简单处理
        pil_image = Image.fromarray(frame)
        
        # 转换为灰度
        gray_image = pil_image.convert('L')
        
        # 创建处理结果，包含帧元数据
        result = {
            'image': np.array(gray_image),
            'timestamp': timestamp,
            'pts': pts,
            'frame_number': self.frame_counters[url],
            'processed_time': time.time()
        }
        
        # 返回处理结果和源URL
        return result, url
    
    def _frame_processed_callback(self, future):
        """
        帧处理完成的回调函数
        
        Args:
            future: 处理结果的Future对象
        """
        try:
            result, url = future.result()
            
            # 如果结果为None，说明帧被跳过
            if result is None:
                return
                
            # 将结果放入处理队列，供异步函数处理
            if not self.processing_queues[url].full():
                self.processing_queues[url].put(result)
        except Exception as e:
            logger.error(f"Error in processing callback: {str(e)}")
    
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
            # 获取处理结果
            result = self.processing_queues[url].get_nowait()
            
            # 执行IO密集型操作，如保存、传输或显示
            await asyncio.get_event_loop().run_in_executor(
                None, self._handle_processed_frame, result, url
            )
            
            self.processing_queues[url].task_done()
    
    def _handle_processed_frame(self, result, url):
        """
        处理已处理帧的函数 - 执行IO操作如保存或显示
        
        Args:
            result: 处理结果字典，包含图像和元数据
            url: 来源RTSP URL
        """
        try:
            # 从结果中提取图像和元数据
            processed_frame = result['image']
            frame_number = result['frame_number']
            timestamp = result['timestamp']
            pts = result['pts']
            processed_time = result['processed_time']
            
            # 计算处理延迟
            processing_delay = processed_time - timestamp
            
            # 日志记录
            logger.debug(f"Handling frame {frame_number} from {url}, delay: {processing_delay:.3f}s")
            
            # 创建PIL图像
            pil_image = Image.fromarray(processed_frame)
            
            # 示例：保存到磁盘（取消注释以启用）
            # 解析URL以获取干净的文件名
            # parsed_url = urlparse(url)
            # safe_url = f"{parsed_url.netloc}{parsed_url.path.replace('/', '_')}"
            # filename = f"output/{safe_url}_{frame_number:06d}.jpg"
            # pil_image.save(filename)
            
            # 在这里添加任何其他IO密集型操作
            pass
        except Exception as e:
            logger.error(f"Error handling processed frame: {str(e)}")


# 示例用法
def main():
    # RTSP流URL列表
    rtsp_urls = [
        "rtsp://username:password@192.168.1.100:554/stream1",
        "rtsp://username:password@192.168.1.101:554/stream1",
        "rtsp://username:password@192.168.1.102:554/stream1",
    ]
    
    # 创建并启动处理器，设置跳帧比例为3（处理1帧，跳过3帧）
    processor = RTSPStreamProcessor(rtsp_urls, frame_skip=3)
    processor.start()
    
    try:
        # 让程序持续运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        # 停止所有处理
        processor.stop()


if __name__ == "__main__":
    main()