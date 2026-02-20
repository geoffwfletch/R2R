import logging
import os
import zipfile
from datetime import datetime
from io import BytesIO
from typing import BinaryIO, Optional
from uuid import UUID

from azure.storage.blob import BlobServiceClient, ContentSettings, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from core.base import FileConfig, FileProvider, R2RException

logger = logging.getLogger()


class AzureBlobFileProvider(FileProvider):
    """Azure Blob Storage implementation of the FileProvider."""

    def __init__(self, config: FileConfig):
        super().__init__(config)

        account_name = self.config.azure_account_name or os.getenv(
            "AZURE_BLOB_ACCOUNT_NAME"
        )
        account_key = self.config.azure_account_key or os.getenv(
            "AZURE_BLOB_ACCOUNT_KEY"
        )
        self.container_name = self.config.azure_container_name or os.getenv(
            "AZURE_BLOB_CONTAINER_NAME"
        )

        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={account_name};"
            f"AccountKey={account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        self.blob_service: BlobServiceClient = (
            BlobServiceClient.from_connection_string(conn_str)
        )
        self._container: Optional[ContainerClient] = None

    @property
    def container(self) -> ContainerClient:
        if self._container is None:
            self._container = self.blob_service.get_container_client(
                self.container_name
            )
        return self._container

    def _blob_path(self, document_id: UUID, file_name: str) -> str:
        return f"ragengine/documents/{document_id}/{file_name}"

    def _preview_path(self, document_id: UUID) -> str:
        return f"ragengine/documents/{document_id}/preview.md"

    def _prefix(self, document_id: UUID) -> str:
        return f"ragengine/documents/{document_id}/"

    async def initialize(self) -> None:
        try:
            self.container.get_container_properties()
            logger.info(
                f"Using existing Azure container: {self.container_name}"
            )
        except ResourceNotFoundError:
            logger.info(
                f"Creating Azure container: {self.container_name}"
            )
            self.blob_service.create_container(self.container_name)

    async def store_file(
        self,
        document_id: UUID,
        file_name: str,
        file_content: BinaryIO,
        file_type: Optional[str] = None,
    ) -> None:
        try:
            blob_name = self._blob_path(document_id, file_name)
            file_content.seek(0)
            blob_client = self.container.get_blob_client(blob_name)
            blob_client.upload_blob(
                file_content.read(),
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=file_type or "application/octet-stream"
                ),
                metadata={
                    "filename": file_name,
                    "document_id": str(document_id),
                },
            )
        except Exception as e:
            logger.error(f"Error storing file in Azure Blob: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to store file in Azure Blob: {e}",
            ) from e

    async def retrieve_file(
        self, document_id: UUID
    ) -> Optional[tuple[str, BinaryIO, int]]:
        prefix = self._prefix(document_id)
        try:
            blobs = list(self.container.list_blobs(name_starts_with=prefix))
            # Find the non-preview file
            target = None
            for blob in blobs:
                if not blob.name.endswith("/preview.md"):
                    target = blob
                    break

            if not target:
                raise R2RException(
                    status_code=404,
                    message=f"File for document {document_id} not found",
                )

            blob_client = self.container.get_blob_client(target.name)
            download = blob_client.download_blob()
            content = BytesIO(download.readall())
            file_name = target.name.split("/")[-1]
            file_size = target.size

            return file_name, content, file_size

        except ResourceNotFoundError:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            ) from None

    async def retrieve_files_as_zip(
        self,
        document_ids: Optional[list[UUID]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[str, BinaryIO, int]:
        if not document_ids:
            raise R2RException(
                status_code=400,
                message="Document IDs must be provided for Azure Blob retrieval",
            )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            for doc_id in document_ids:
                try:
                    result = await self.retrieve_file(doc_id)
                    if result:
                        file_name, file_content, _ = result
                        file_content.seek(0)
                        zip_file.writestr(file_name, file_content.read())
                except R2RException as e:
                    if e.status_code == 404:
                        continue
                    raise

        zip_buffer.seek(0)
        zip_size = zip_buffer.getbuffer().nbytes
        if zip_size == 0:
            raise R2RException(
                status_code=404,
                message="No files found for the specified document IDs",
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"files_export_{timestamp}.zip", zip_buffer, zip_size

    async def delete_file(self, document_id: UUID) -> bool:
        prefix = self._prefix(document_id)
        blobs = list(self.container.list_blobs(name_starts_with=prefix))
        # Filter out preview to check if actual file exists
        file_blobs = [
            b for b in blobs if not b.name.endswith("/preview.md")
        ]
        if not file_blobs:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            )

        # Delete all blobs including preview
        for blob in blobs:
            self.container.delete_blob(blob.name)
        return True

    async def get_files_overview(
        self,
        offset: int,
        limit: int,
        filter_document_ids: Optional[list[UUID]] = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> list[dict]:
        results = []

        if filter_document_ids:
            for doc_id in filter_document_ids:
                prefix = self._prefix(doc_id)
                blobs = list(
                    self.container.list_blobs(name_starts_with=prefix)
                )
                for blob in blobs:
                    if blob.name.endswith("/preview.md"):
                        continue
                    file_name = blob.name.split("/")[-1]
                    if filter_file_names and file_name not in filter_file_names:
                        continue
                    results.append(
                        {
                            "document_id": doc_id,
                            "file_name": file_name,
                            "file_key": blob.name,
                            "file_size": blob.size,
                            "file_type": blob.content_settings.content_type
                            if blob.content_settings
                            else None,
                            "created_at": blob.creation_time,
                            "updated_at": blob.last_modified,
                        }
                    )
        else:
            prefix = "ragengine/documents/"
            blobs = list(
                self.container.list_blobs(name_starts_with=prefix)
            )
            # Skip preview files, apply pagination
            filtered = [
                b for b in blobs if not b.name.endswith("/preview.md")
            ]
            page = filtered[offset : offset + limit]
            for blob in page:
                parts = blob.name.split("/")
                try:
                    doc_id = UUID(parts[2])
                except (ValueError, IndexError):
                    continue
                file_name = parts[-1]
                if filter_file_names and file_name not in filter_file_names:
                    continue
                results.append(
                    {
                        "document_id": doc_id,
                        "file_name": file_name,
                        "file_key": blob.name,
                        "file_size": blob.size,
                        "file_type": blob.content_settings.content_type
                        if blob.content_settings
                        else None,
                        "created_at": blob.creation_time,
                        "updated_at": blob.last_modified,
                    }
                )

        if not results:
            raise R2RException(
                status_code=404,
                message="No files found with the given filters",
            )
        return results

    async def store_markdown_preview(
        self,
        document_id: UUID,
        file_name: str,
        markdown_content: str,
    ) -> None:
        try:
            blob_name = self._preview_path(document_id)
            blob_client = self.container.get_blob_client(blob_name)
            blob_client.upload_blob(
                markdown_content.encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(
                    content_type="text/markdown"
                ),
                metadata={"document_id": str(document_id)},
            )
        except Exception as e:
            logger.error(f"Error storing markdown preview in Azure: {e}")

    async def retrieve_markdown_preview(
        self, document_id: UUID
    ) -> Optional[str]:
        try:
            blob_name = self._preview_path(document_id)
            blob_client = self.container.get_blob_client(blob_name)
            download = blob_client.download_blob()
            return download.readall().decode("utf-8")
        except ResourceNotFoundError:
            return None
