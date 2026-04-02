import oss2
import time
import os
from typing import Optional


class OSSUploader:
    """阿里云OSS上传工具类（配置内置）"""

    # === 内置配置 ===
    ACCESS_KEY_ID = ''
    ACCESS_KEY_SECRET = ''
    BUCKET_NAME = 'ai-badou'
    ENDPOINT = 'https://oss-cn-beijing.aliyuncs.com'

    def __init__(self):
        """初始化OSS上传器，直接使用内置配置"""
        self.auth = oss2.Auth(self.ACCESS_KEY_ID, self.ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(self.auth, self.ENDPOINT, self.BUCKET_NAME)
        self.bucket_name = self.BUCKET_NAME
        self.endpoint = self.ENDPOINT
        print("✓ OSS上传器初始化成功")

    def upload_file(self, local_file_path: str, oss_object_name: str = None,
                    is_public_read: bool = True, url_expire_seconds: int = 3600) -> Optional[str]:
        """
        上传文件到OSS并返回URL

        Args:
            local_file_path: 本地文件路径
            oss_object_name: OSS中的对象名称（如果为None，自动生成）
            is_public_read: 是否使用公共读URL（需要bucket设置为公共读）
            url_expire_seconds: 签名URL有效期（秒），仅当is_public_read=False时生效

        Returns:
            文件的访问URL，失败返回None
        """
        try:
            # 如果没有指定OSS对象名称，自动生成
            if oss_object_name is None:
                timestamp = int(time.time())
                file_ext = os.path.splitext(local_file_path)[1]
                oss_object_name = f"uploads/{timestamp}_{os.path.basename(local_file_path)}"

            # 上传文件
            self.bucket.put_object_from_file(oss_object_name, local_file_path)
            print(f"✓ 文件上传成功: {oss_object_name}")

            # 生成访问URL
            if is_public_read:
                # 公共读URL
                url = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{oss_object_name}"
            else:
                # 签名URL
                url = self.bucket.sign_url('GET', oss_object_name, url_expire_seconds)

            return url

        except oss2.exceptions.OssError as e:
            print(f"✗ 上传失败: {e}")
            return None

    def upload_image(self, image_path: str, prefix: str = "images") -> Optional[str]:
        """
        专门上传图片的方法

        Args:
            image_path: 图片本地路径
            prefix: OSS中的目录前缀

        Returns:
            图片的访问URL
        """
        timestamp = int(time.time())
        file_ext = os.path.splitext(image_path)[1]
        oss_object_name = f"{prefix}/page_{timestamp}{file_ext}"
        return self.upload_file(image_path, oss_object_name)

    def delete_file(self, oss_object_name: str) -> bool:
        """
        删除OSS中的文件

        Args:
            oss_object_name: OSS中的对象名称

        Returns:
            是否删除成功
        """
        try:
            self.bucket.delete_object(oss_object_name)
            print(f"✓ 文件删除成功: {oss_object_name}")
            return True
        except oss2.exceptions.OssError as e:
            print(f"✗ 删除失败: {e}")
            return False