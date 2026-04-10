"""
Storage Manager - HDFS-Compatible Storage Abstraction Layer
Provides a local filesystem implementation that mirrors HDFS directory conventions.
Easily swappable to real HDFS for production deployment.
"""

import os
import shutil
import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


class StorageManager:
    """
    Abstract storage manager interface.
    Implements data lake zones: raw, processed, models, reports.
    """

    ZONES = ["raw", "processed", "models", "reports"]

    def __init__(self, config: dict):
        self.config = config
        self.data_root = config.get("data_root", "data")

    def store(self, data: Any, zone: str, filename: str) -> str:
        raise NotImplementedError

    def retrieve(self, zone: str, filename: str) -> Any:
        raise NotImplementedError

    def list_files(self, zone: str, pattern: str = "*") -> List[str]:
        raise NotImplementedError

    def get_metadata(self, zone: str, filename: str) -> Dict:
        raise NotImplementedError

    def delete(self, zone: str, filename: str) -> bool:
        raise NotImplementedError


class LocalStorageManager(StorageManager):
    """
    Local filesystem storage manager implementing HDFS-like data lake structure.

    Directory Structure:
        data/
        |---- raw/          # Landing zone - raw ingested log files
        |---- processed/    # Cleaned and transformed data
        |---- models/       # Trained ML models and artifacts
        |---- reports/      # Generated analytical reports
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._ensure_zones()

    def _ensure_zones(self):
        """Create data lake zone directories if they don't exist."""
        for zone in self.ZONES:
            zone_path = os.path.join(self.data_root, zone)
            os.makedirs(zone_path, exist_ok=True)

    def _zone_path(self, zone: str, filename: str = "") -> str:
        """Get the full path for a file in a zone."""
        if zone not in self.ZONES:
            raise ValueError(f"Invalid zone: {zone}. Valid zones: {self.ZONES}")
        return os.path.join(self.data_root, zone, filename)

    def store(self, data: Any, zone: str, filename: str) -> str:
        """
        Store data to a specific zone.

        Args:
            data: Content to store (str, bytes, dict, or list)
            zone: Data lake zone (raw, processed, models, reports)
            filename: Target filename

        Returns:
            Full path to the stored file
        """
        filepath = self._zone_path(zone, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if isinstance(data, dict) or isinstance(data, list):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, bytes):
            with open(filepath, "wb") as f:
                f.write(data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))

        # Store metadata
        self._write_metadata(zone, filename, filepath)
        return filepath

    def store_file(self, source_path: str, zone: str, filename: Optional[str] = None) -> str:
        """Copy an existing file into a data lake zone."""
        if filename is None:
            filename = os.path.basename(source_path)
        dest = self._zone_path(zone, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(source_path, dest)
        self._write_metadata(zone, filename, dest)
        return dest

    def retrieve(self, zone: str, filename: str) -> str:
        """Retrieve file content from a zone."""
        filepath = self._zone_path(zone, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def retrieve_json(self, zone: str, filename: str) -> Any:
        """Retrieve and parse JSON file from a zone."""
        content = self.retrieve(zone, filename)
        return json.loads(content)

    def list_files(self, zone: str, pattern: str = "*") -> List[str]:
        """List files in a zone, optionally filtered by glob pattern."""
        zone_dir = self._zone_path(zone)
        if not os.path.exists(zone_dir):
            return []

        from pathlib import Path
        p = Path(zone_dir)

        if pattern == "*":
            files = [f.name for f in p.iterdir() if f.is_file() and not f.name.startswith("_")]
        else:
            files = [f.name for f in p.glob(pattern) if f.is_file() and not f.name.startswith("_")]

        return sorted(files)

    def get_metadata(self, zone: str, filename: str) -> Dict:
        """Get metadata for a stored file."""
        meta_path = self._zone_path(zone, f"_metadata/{filename}.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)

        # Fall back to file system metadata
        filepath = self._zone_path(zone, filename)
        if os.path.exists(filepath):
            stat = os.stat(filepath)
            return {
                "filename": filename,
                "zone": zone,
                "size_bytes": stat.st_size,
                "size_human": self._human_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        return {}

    def delete(self, zone: str, filename: str) -> bool:
        """Delete a file from a zone."""
        filepath = self._zone_path(zone, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            # Clean up metadata
            meta_path = self._zone_path(zone, f"_metadata/{filename}.meta.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
            return True
        return False

    def get_zone_stats(self, zone: str) -> Dict:
        """Get aggregate statistics for a zone."""
        zone_dir = self._zone_path(zone)
        if not os.path.exists(zone_dir):
            return {"file_count": 0, "total_size": 0}

        files = self.list_files(zone)
        total_size = sum(
            os.path.getsize(self._zone_path(zone, f))
            for f in files if os.path.exists(self._zone_path(zone, f))
        )

        return {
            "zone": zone,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_human": self._human_size(total_size),
            "files": files,
        }

    def get_all_stats(self) -> Dict:
        """Get statistics for all zones."""
        return {zone: self.get_zone_stats(zone) for zone in self.ZONES}

    def _write_metadata(self, zone: str, filename: str, filepath: str):
        """Write metadata file for a stored file."""
        meta_dir = self._zone_path(zone, "_metadata")
        os.makedirs(meta_dir, exist_ok=True)

        stat = os.stat(filepath)
        md5 = self._compute_md5(filepath)

        metadata = {
            "filename": filename,
            "zone": zone,
            "path": filepath,
            "size_bytes": stat.st_size,
            "size_human": self._human_size(stat.st_size),
            "md5": md5,
            "ingested_at": datetime.now().isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

        meta_path = os.path.join(meta_dir, f"{filename}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _compute_md5(filepath: str, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


class HDFSStorageManager(StorageManager):
    """
    HDFS storage manager for production Hadoop deployments.
    Uses PyArrow's HDFS interface for cluster communication.

    Requirements:
        pip install pyarrow

    Configuration:
        storage:
            backend: hdfs
            data_root: /data/log_analyzer
            hdfs:
                host: namenode-host
                port: 8020
                user: hadoop
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.hdfs_config = config.get("hdfs", {})
        self.host = self.hdfs_config.get("host", "localhost")
        self.port = self.hdfs_config.get("port", 8020)
        self.user = self.hdfs_config.get("user", "hadoop")
        self.fs = None
        self._connect()
        self._ensure_zones()

    def _connect(self):
        """Establish connection to HDFS NameNode via PyArrow."""
        try:
            import pyarrow.fs as pafs
            self.fs = pafs.HadoopFileSystem(
                host=self.host,
                port=self.port,
                user=self.user,
            )
            print(f"  [OK] Connected to HDFS at {self.host}:{self.port}")
        except ImportError:
            raise ImportError(
                "PyArrow is required for HDFS backend. "
                "Install with: pip install pyarrow"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to HDFS at {self.host}:{self.port}: {e}"
            )

    def _ensure_zones(self):
        """Create data lake zone directories on HDFS."""
        for zone in self.ZONES:
            zone_path = f"{self.data_root}/{zone}"
            try:
                self.fs.create_dir(zone_path, recursive=True)
            except Exception:
                pass  # Directory may already exist

    def _zone_path(self, zone: str, filename: str = "") -> str:
        if zone not in self.ZONES:
            raise ValueError(f"Invalid zone: {zone}. Valid zones: {self.ZONES}")
        path = f"{self.data_root}/{zone}"
        if filename:
            path = f"{path}/{filename}"
        return path

    def store(self, data: Any, zone: str, filename: str) -> str:
        """
        Store data to an HDFS zone.

        Args:
            data: Content to store (str, bytes, dict, or list)
            zone: Data lake zone
            filename: Target filename

        Returns:
            HDFS path to the stored file
        """
        filepath = self._zone_path(zone, filename)

        if isinstance(data, (dict, list)):
            content = json.dumps(data, indent=2, default=str).encode("utf-8")
        elif isinstance(data, bytes):
            content = data
        else:
            content = str(data).encode("utf-8")

        with self.fs.open_output_stream(filepath) as f:
            f.write(content)

        self._write_metadata(zone, filename, filepath)
        return filepath

    def store_file(self, source_path: str, zone: str, filename: Optional[str] = None) -> str:
        """Upload a local file to an HDFS zone."""
        if filename is None:
            filename = os.path.basename(source_path)
        dest = self._zone_path(zone, filename)

        # Read local file and write to HDFS
        with open(source_path, "rb") as local_f:
            content = local_f.read()
        with self.fs.open_output_stream(dest) as hdfs_f:
            hdfs_f.write(content)

        self._write_metadata(zone, filename, dest)
        return dest

    def retrieve(self, zone: str, filename: str) -> str:
        """Retrieve file content from HDFS."""
        filepath = self._zone_path(zone, filename)
        try:
            with self.fs.open_input_stream(filepath) as f:
                return f.read().decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"HDFS file not found: {filepath}")

    def retrieve_json(self, zone: str, filename: str) -> Any:
        """Retrieve and parse JSON file from HDFS."""
        content = self.retrieve(zone, filename)
        return json.loads(content)

    def list_files(self, zone: str, pattern: str = "*") -> List[str]:
        """List files in an HDFS zone."""
        zone_dir = self._zone_path(zone)
        try:
            file_infos = self.fs.get_file_info(
                self.fs.get_file_info(zone_dir).path
            )
            # List directory contents
            from pyarrow.fs import FileSelector
            selector = FileSelector(zone_dir, recursive=False)
            entries = self.fs.get_file_info(selector)
            files = [
                os.path.basename(entry.path)
                for entry in entries
                if entry.is_file and not os.path.basename(entry.path).startswith("_")
            ]
            return sorted(files)
        except Exception:
            return []

    def get_metadata(self, zone: str, filename: str) -> Dict:
        """Get metadata for a file on HDFS."""
        filepath = self._zone_path(zone, filename)
        try:
            info = self.fs.get_file_info(filepath)
            return {
                "filename": filename,
                "zone": zone,
                "hdfs_path": filepath,
                "size_bytes": info.size,
                "size_human": self._human_size(info.size) if info.size else "0 B",
                "modified": info.mtime.isoformat() if info.mtime else None,
                "type": str(info.type),
            }
        except Exception:
            return {}

    def delete(self, zone: str, filename: str) -> bool:
        """Delete a file from HDFS."""
        filepath = self._zone_path(zone, filename)
        try:
            self.fs.delete_file(filepath)
            meta_path = self._zone_path(zone, f"_metadata/{filename}.meta.json")
            try:
                self.fs.delete_file(meta_path)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def get_zone_stats(self, zone: str) -> Dict:
        """Get aggregate statistics for an HDFS zone."""
        files = self.list_files(zone)
        total_size = 0
        for f in files:
            try:
                info = self.fs.get_file_info(self._zone_path(zone, f))
                total_size += info.size or 0
            except Exception:
                pass

        return {
            "zone": zone,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_human": self._human_size(total_size),
            "files": files,
        }

    def _write_metadata(self, zone: str, filename: str, filepath: str):
        """Write metadata JSON alongside the file on HDFS."""
        meta_dir = self._zone_path(zone, "_metadata")
        try:
            self.fs.create_dir(meta_dir, recursive=True)
        except Exception:
            pass

        try:
            info = self.fs.get_file_info(filepath)
            metadata = {
                "filename": filename,
                "zone": zone,
                "hdfs_path": filepath,
                "size_bytes": info.size,
                "ingested_at": datetime.now().isoformat(),
                "modified_at": info.mtime.isoformat() if info.mtime else None,
            }
            meta_content = json.dumps(metadata, indent=2).encode("utf-8")
            meta_path = f"{meta_dir}/{filename}.meta.json"
            with self.fs.open_output_stream(meta_path) as f:
                f.write(meta_content)
        except Exception:
            pass  # Metadata is best-effort

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


def get_storage_manager(config: dict) -> StorageManager:
    """
    Factory function to create the appropriate storage manager.

    Args:
        config: Storage configuration dict with 'backend' key.
                'local' -> LocalStorageManager (default)
                'hdfs'  -> HDFSStorageManager (requires pyarrow)

    Returns:
        StorageManager instance
    """
    backend = config.get("backend", "local")
    if backend == "local":
        return LocalStorageManager(config)
    elif backend == "hdfs":
        return HDFSStorageManager(config)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")

