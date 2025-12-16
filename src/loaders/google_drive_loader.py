"""Google Drive document loader with OAuth2 authentication."""
import io
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.loaders.base import BaseLoader, Document
from src.loaders.factory import register_loader

logger = logging.getLogger(__name__)

# Scopes for Google Drive access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# MIME type mappings
GOOGLE_MIME_TYPES = {
    'application/vnd.google-apps.document': {
        'export_mime': 'text/plain',
        'extension': '.txt',
    },
    'application/vnd.google-apps.spreadsheet': {
        'export_mime': 'text/csv',
        'extension': '.csv',
    },
    'application/vnd.google-apps.presentation': {
        'export_mime': 'text/plain',
        'extension': '.txt',
    },
}

SUPPORTED_MIME_TYPES = [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'text/html',
    'application/vnd.google-apps.document',
    'application/vnd.google-apps.spreadsheet',
    'application/vnd.google-apps.presentation',
]


@register_loader('.gdrive')
class GoogleDriveLoader(BaseLoader):
    """
    Load documents from Google Drive using OAuth2.
    
    Features:
    - OAuth2 user consent (client authorizes their own Drive)
    - Supports Google Docs, Sheets, Slides (exports to text)
    - Supports PDF, TXT, MD, HTML files
    - Folder traversal (recursive)
    - Token caching (re-auth not needed every time)
    
    Usage:
        loader = GoogleDriveLoader(
            credentials_path="config/google_credentials.json",
            token_path="config/google_token.json",
        )
        
        # Load specific file
        docs = loader.load_file(file_id="1abc...")
        
        # Load entire folder
        docs = loader.load_folder(folder_id="1xyz...")
        
        # Search and load
        docs = loader.search_and_load(query="quarterly report")
    """
    
    def __init__(
        self,
        credentials_path: str = "config/google_credentials.json",
        token_path: str = "config/google_token.json",
    ):
        """
        Initialize Google Drive loader.
        
        Args:
            credentials_path: Path to OAuth2 credentials JSON
            token_path: Path to store/load cached token
        """
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self._service = None
        
        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_path}\n"
                "Download from Google Cloud Console → APIs & Services → Credentials"
            )
        
        logger.info("Initialized GoogleDriveLoader")
    
    @property
    def service(self):
        """Get or create authenticated Drive service."""
        if self._service is None:
            self._service = self._authenticate()
        return self._service
    
    def _authenticate(self):
        """Authenticate with Google Drive using OAuth2."""
        creds = None
        
        # Load cached token if exists
        if self.token_path.exists():
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired token...")
                creds.refresh(Request())
            else:
                logger.info("Starting OAuth2 flow...")
                flow = Flow.from_client_secrets_file(
                    str(self.credentials_path),
                    scopes=SCOPES,
                    redirect_uri='http://localhost:8080/'
                )
                
                # Generate auth URL
                auth_url, _ = flow.authorization_url(prompt='consent')
                
                print("\n" + "="*60)
                print("GOOGLE DRIVE AUTHORIZATION")
                print("="*60)
                print("\n1. Open this URL in your browser:\n")
                print(auth_url)
                print("\n2. Sign in and authorize the application")
                print("3. You'll be redirected to localhost (may show error page)")
                print("4. Copy the FULL URL from your browser address bar")
                print("   (It will look like: http://localhost:8080/?code=...)")
                print("\n")
                
                redirect_response = input("Paste the full redirect URL here: ").strip()
                
                # Exchange code for token
                flow.fetch_token(authorization_response=redirect_response)
                creds = flow.credentials
            
            # Save token for next time
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
            logger.info(f"Token saved to {self.token_path}")
        
        service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive authenticated successfully")
        return service
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load from Google Drive file ID.
        
        Args:
            file_path: Google Drive file ID
            
        Returns:
            List containing single Document
        """
        return self.load_file(file_id=file_path)
    
    def load_file(self, file_id: str) -> List[Document]:
        """
        Load a single file by ID.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            List containing single Document
        """
        # Get file metadata
        file_meta = self.service.files().get(
            fileId=file_id,
            fields='id, name, mimeType, modifiedTime, owners, webViewLink'
        ).execute()
        
        content = self._download_file(file_id, file_meta['mimeType'])
        
        if content is None:
            logger.warning(f"Could not download file: {file_meta['name']}")
            return []
        
        metadata = {
            'source': f"gdrive://{file_id}",
            'file_name': file_meta['name'],
            'mime_type': file_meta['mimeType'],
            'modified_time': file_meta.get('modifiedTime'),
            'web_link': file_meta.get('webViewLink'),
            'drive_id': file_id,
        }
        
        if 'owners' in file_meta and file_meta['owners']:
            metadata['owner'] = file_meta['owners'][0].get('displayName')
        
        return [Document(content=content, metadata=metadata)]
    
    def load_folder(
        self,
        folder_id: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load all documents from a folder.
        
        Args:
            folder_id: Google Drive folder ID
            recursive: Include subfolders
            file_types: Filter by MIME types (None = all supported)
            
        Returns:
            List of Documents
        """
        documents = []
        self._load_folder_recursive(folder_id, documents, recursive, file_types)
        logger.info(f"Loaded {len(documents)} documents from folder")
        return documents
    
    def _load_folder_recursive(
        self,
        folder_id: str,
        documents: List[Document],
        recursive: bool,
        file_types: Optional[List[str]],
    ):
        """Recursively load files from folder."""
        query = f"'{folder_id}' in parents and trashed = false"
        
        page_token = None
        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)',
                pageToken=page_token,
            ).execute()
            
            for file in response.get('files', []):
                mime_type = file['mimeType']
                
                # Handle folders
                if mime_type == 'application/vnd.google-apps.folder':
                    if recursive:
                        self._load_folder_recursive(
                            file['id'], documents, recursive, file_types
                        )
                    continue
                
                # Check if supported
                if mime_type not in SUPPORTED_MIME_TYPES:
                    logger.debug(f"Skipping unsupported type: {file['name']} ({mime_type})")
                    continue
                
                # Filter by type if specified
                if file_types and mime_type not in file_types:
                    continue
                
                # Download file
                content = self._download_file(file['id'], mime_type)
                if content:
                    documents.append(Document(
                        content=content,
                        metadata={
                            'source': f"gdrive://{file['id']}",
                            'file_name': file['name'],
                            'mime_type': mime_type,
                            'modified_time': file.get('modifiedTime'),
                            'web_link': file.get('webViewLink'),
                            'drive_id': file['id'],
                        }
                    ))
            
            page_token = response.get('nextPageToken')
            if not page_token:
                break
    
    def _download_file(self, file_id: str, mime_type: str) -> Optional[str]:
        """Download file content as text."""
        try:
            # Google Workspace files need export
            if mime_type in GOOGLE_MIME_TYPES:
                export_mime = GOOGLE_MIME_TYPES[mime_type]['export_mime']
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime
                )
            else:
                request = self.service.files().get_media(fileId=file_id)
            
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            content = buffer.getvalue()
            
            # Handle PDFs
            if mime_type == 'application/pdf':
                return self._extract_pdf_text(content)
            
            # Decode text
            return content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None
    
    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def search_and_load(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Document]:
        """
        Search Drive and load matching documents.
        
        Args:
            query: Search query (uses Drive search syntax)
            max_results: Maximum files to load
            
        Returns:
            List of Documents
        """
        # Build search query
        search_query = f"fullText contains '{query}' and trashed = false"
        
        response = self.service.files().list(
            q=search_query,
            spaces='drive',
            fields='files(id, name, mimeType)',
            pageSize=max_results,
        ).execute()
        
        documents = []
        for file in response.get('files', []):
            if file['mimeType'] in SUPPORTED_MIME_TYPES:
                docs = self.load_file(file['id'])
                documents.extend(docs)
        
        logger.info(f"Search '{query}' returned {len(documents)} documents")
        return documents
    
    def list_files(
        self,
        folder_id: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List files (without downloading).
        
        Args:
            folder_id: Folder to list (None = root)
            max_results: Maximum files to return
            
        Returns:
            List of file metadata dicts
        """
        query = "trashed = false"
        if folder_id:
            query = f"'{folder_id}' in parents and " + query
        
        response = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType, modifiedTime, size)',
            pageSize=max_results,
        ).execute()
        
        return response.get('files', [])
    
    def supported_extensions(self) -> List[str]:
        """Return list of supported extensions."""
        return [".gdrive"]  # Virtual extension for factory registration
    
    def health_check(self) -> bool:
        """Check if Google Drive is accessible."""
        try:
            self.service.files().list(pageSize=1).execute()
            return True
        except Exception as e:
            logger.error(f"Google Drive health check failed: {e}")
            return False
