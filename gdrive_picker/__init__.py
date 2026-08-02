"""
A minimal, hand-rolled Streamlit custom component that opens Google's
official Picker widget so users can attach a PDF straight from Google
Drive.

Unlike the plain browser <input type="file"> flow, this downloads the
file's bytes directly from the Drive API over a normal HTTPS request,
instead of going through the phone/browser's Storage Access Framework
file-stream -- which is what was causing valid PDFs picked from Drive's
cloud folder view to show up truncated or rejected.

No npm/React build is required: the frontend is a single static HTML
file that speaks Streamlit's component postMessage protocol directly.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_gdrive_picker_component = components.declare_component(
    "gdrive_picker",
    path=_COMPONENT_DIR,
)


def gdrive_pdf_picker(client_id, api_key, key=None):
    """
    Renders an "Import PDF from Google Drive" button.

    Args:
        client_id: OAuth 2.0 Web Client ID from Google Cloud Console.
        api_key: API key (restricted to the Picker API).
        key: Optional Streamlit widget key.

    Returns:
        None until the user picks a file. Once a file is picked and
        downloaded, returns a dict: {"name": <filename>, "data": <base64
        str of the raw file bytes>}. The dict is returned on the rerun
        that follows a successful pick -- check for it and reset/consume
        it the same way you would a file_uploader result.
    """
    return _gdrive_picker_component(
        client_id=client_id,
        api_key=api_key,
        key=key,
        default=None,
    )