# app.py - Python PDF Text Extraction Service
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import requests
import io
import logging
import time
from typing import Optional, Dict, Any
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for your Node.js app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self):
        self.methods = [
            ("pdfplumber", self._extract_with_pdfplumber),
            ("pymupdf", self._extract_with_pymupdf),
            ("pypdf2", self._extract_with_pypdf2)
        ]
    
    def _extract_with_pdfplumber(self, pdf_buffer: io.BytesIO) -> str:
        """Extract text using pdfplumber (most reliable)"""
        text = ""
        try:
            with pdfplumber.open(pdf_buffer) as pdf:
                logger.info(f"📄 PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text() or ""
                        text += f"Page {i}:\n{page_text}\n\n"
                        logger.info(f"📄 Extracted {len(page_text)} chars from page {i}")
                    except Exception as page_error:
                        logger.warning(f"⚠️ Failed to extract from page {i}: {page_error}")
                        continue
        except Exception as e:
            raise Exception(f"pdfplumber extraction failed: {str(e)}")
        
        return text.strip()
    
    def _extract_with_pymupdf(self, pdf_buffer: io.BytesIO) -> str:
        """Extract text using PyMuPDF (good fallback)"""
        text = ""
        try:
            pdf_buffer.seek(0)
            doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
            logger.info(f"📄 PDF has {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
                    logger.info(f"📄 Extracted {len(page_text)} chars from page {page_num + 1}")
                except Exception as page_error:
                    logger.warning(f"⚠️ Failed to extract from page {page_num + 1}: {page_error}")
                    continue
            
            doc.close()
        except Exception as e:
            raise Exception(f"PyMuPDF extraction failed: {str(e)}")
        
        return text.strip()
    
    def _extract_with_pypdf2(self, pdf_buffer: io.BytesIO) -> str:
        """Extract text using PyPDF2 (last resort)"""
        text = ""
        try:
            pdf_buffer.seek(0)
            reader = PyPDF2.PdfReader(pdf_buffer)
            logger.info(f"📄 PDF has {len(reader.pages)} pages")
            
            for i, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    text += f"Page {i}:\n{page_text}\n\n"
                    logger.info(f"📄 Extracted {len(page_text)} chars from page {i}")
                except Exception as page_error:
                    logger.warning(f"⚠️ Failed to extract from page {i}: {page_error}")
                    continue
        except Exception as e:
            raise Exception(f"PyPDF2 extraction failed: {str(e)}")
        
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text for better processing"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters but keep common symbols
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive empty lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate extraction confidence score"""
        if not text:
            return 0.0
        
        confidence = 0.3  # Base confidence
        
        # Length indicators
        if len(text) > 50:
            confidence += 0.1
        if len(text) > 200:
            confidence += 0.1
        if len(text) > 500:
            confidence += 0.1
        
        # Content quality indicators for invoices
        has_numbers = bool(re.search(r'\d', text))
        has_currency = bool(re.search(r'[\$€£¥₹]|\b(?:USD|EUR|GBP|INR)\b', text, re.I))
        has_date = bool(re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}', text))
        has_invoice_terms = bool(re.search(r'\b(?:invoice|bill|receipt|total|amount|vendor|supplier|customer|tax|qty|quantity|price|date)\b', text, re.I))
        
        if has_numbers:
            confidence += 0.1
        if has_currency:
            confidence += 0.15
        if has_date:
            confidence += 0.1
        if has_invoice_terms:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def extract_text(self, pdf_buffer: io.BytesIO) -> Dict[str, Any]:
        """Extract text with multiple fallback methods"""
        start_time = time.time()
        
        for method_name, method_func in self.methods:
            try:
                logger.info(f"🔧 Attempting extraction with {method_name}...")
                pdf_buffer.seek(0)  # Reset buffer position
                
                text = method_func(pdf_buffer)
                
                if text and len(text.strip()) > 10:
                    cleaned_text = self._clean_text(text)
                    confidence = self._calculate_confidence(cleaned_text)
                    processing_time = time.time() - start_time
                    
                    logger.info(f"✅ {method_name} extraction successful: {len(cleaned_text)} chars, confidence: {confidence:.2f}")
                    
                    return {
                        "success": True,
                        "text": cleaned_text,
                        "method": method_name,
                        "confidence": confidence,
                        "processing_time_ms": int(processing_time * 1000),
                        "text_length": len(cleaned_text),
                        "sample": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
                    }
                else:
                    logger.warning(f"⚠️ {method_name} returned insufficient text")
                    
            except Exception as e:
                logger.error(f"❌ {method_name} extraction failed: {str(e)}")
                continue
        
        # If all methods failed
        processing_time = time.time() - start_time
        return {
            "success": False,
            "error": "All PDF extraction methods failed",
            "processing_time_ms": int(processing_time * 1000)
        }

# Initialize extractor
extractor = PDFTextExtractor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "PDF Text Extraction Service",
        "version": "1.0.0",
        "methods": ["pdfplumber", "pymupdf", "pypdf2"]
    })

@app.route('/extract', methods=['POST'])
def extract_text():
    """Extract text from PDF - accepts URL or file upload"""
    try:
        # Check if PDF URL is provided
        if request.json and 'fileUrl' in request.json:
            file_url = request.json['fileUrl']
            logger.info(f"📄 Extracting from URL: {file_url[:100]}...")
            
            # Download PDF from URL
            response = requests.get(file_url, timeout=30)
            if not response.ok:
                return jsonify({
                    "success": False,
                    "error": f"Failed to download PDF: {response.status_code}"
                }), 400
            
            pdf_buffer = io.BytesIO(response.content)
            
        # Check if PDF file is uploaded
        elif 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No file provided"
                }), 400
            
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({
                    "success": False,
                    "error": "File must be a PDF"
                }), 400
            
            pdf_buffer = io.BytesIO(file.read())
            logger.info(f"📄 Extracting from uploaded file: {file.filename}")
            
        else:
            return jsonify({
                "success": False,
                "error": "Either 'fileUrl' in JSON body or 'file' upload is required"
            }), 400
        
        # Validate PDF buffer
        pdf_buffer.seek(0)
        if len(pdf_buffer.getvalue()) == 0:
            return jsonify({
                "success": False,
                "error": "PDF file is empty"
            }), 400
        
        # Check file size (25MB limit)
        if len(pdf_buffer.getvalue()) > 25 * 1024 * 1024:
            return jsonify({
                "success": False,
                "error": "PDF file too large (>25MB)"
            }), 400
        
        # Validate PDF header
        pdf_buffer.seek(0)
        header = pdf_buffer.read(4)
        if header != b'%PDF':
            return jsonify({
                "success": False,
                "error": "Invalid PDF file format"
            }), 400
        
        pdf_buffer.seek(0)  # Reset for extraction
        
        logger.info(f"📊 PDF buffer size: {len(pdf_buffer.getvalue()) / (1024*1024):.2f} MB")
        
        # Extract text
        result = extractor.extract_text(pdf_buffer)
        
        if result["success"]:
            logger.info(f"🎉 PDF extraction completed successfully in {result['processing_time_ms']}ms")
            return jsonify(result)
        else:
            logger.error(f"❌ PDF extraction failed: {result.get('error')}")
            return jsonify(result), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Network error: {str(e)}"
        }), 500
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/extract-batch', methods=['POST'])
def extract_batch():
    """Extract text from multiple PDFs"""
    try:
        if not request.json or 'fileUrls' not in request.json:
            return jsonify({
                "success": False,
                "error": "fileUrls array is required"
            }), 400
        
        file_urls = request.json['fileUrls']
        if not isinstance(file_urls, list) or len(file_urls) == 0:
            return jsonify({
                "success": False,
                "error": "fileUrls must be a non-empty array"
            }), 400
        
        if len(file_urls) > 10:  # Limit batch size
            return jsonify({
                "success": False,
                "error": "Maximum 10 files allowed per batch"
            }), 400
        
        results = []
        for i, file_url in enumerate(file_urls):
            try:
                logger.info(f"📄 Processing file {i+1}/{len(file_urls)}: {file_url[:100]}...")
                
                response = requests.get(file_url, timeout=30)
                if not response.ok:
                    results.append({
                        "fileUrl": file_url,
                        "success": False,
                        "error": f"Failed to download: {response.status_code}"
                    })
                    continue
                
                pdf_buffer = io.BytesIO(response.content)
                result = extractor.extract_text(pdf_buffer)
                result["fileUrl"] = file_url
                results.append(result)
                
            except Exception as e:
                results.append({
                    "fileUrl": file_url,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(file_urls),
            "successful": sum(1 for r in results if r.get("success", False))
        })
        
    except Exception as e:
        logger.error(f"❌ Batch processing error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Batch processing error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)