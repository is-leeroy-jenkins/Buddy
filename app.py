'''
  ******************************************************************************************
      Assembly:                Wario
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 01-31-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        01-20-2026
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

	     app.py
	     Copyright ©  2026  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com

  </copyright>
  <summary>
    app.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
from boogr import Error
import config as cfg
import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance
import base64
import io
from pathlib import Path
import multiprocessing
import os
import sqlite3
from typing import Any, Dict, List, Tuple, Optional
import tempfile
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
from gpt import (
	Chat,
	Images,
	Embeddings,
	TTS,
	Transcription,
	Translation,
	VectorStore,
)

from gemini import ( Chat, Images, Embeddings, Transcription, TTS, Translation, FileStore )

from grok import ( Chat, Images, Embeddings, Files, Collections )

# ==============================================================================
# CONSTANTS
# ==============================================================================
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )

FAVICON = r'resources/favicon.ico'

CRS = r'https://www.congress.gov/crs-appropriations-status-table'

GPT_LOGO = r'resources/buddy_logo.ico'

GEMINI_LOGO = r'resources/gemini_logo.png'

GROQ_LOGO = r'resources/grok_logo.png'

GPT_MODES = [ 'Chat',
              'Text',
              'Images',
              'Audio',
              'Embeddings',
              'Documents',
              'Files',
              'VectorStore' ]

GEMINI_MODES = [ 'Chat',
                 'Text',
                 'Images',
                 'Audio',
                 'Embeddings',
                 'Documents',
                 'Files' ]

GROQ_MODES = [ 'Chat',
               'Text',
               'Images',
               'Embeddings',
               'Documents',
               'Files',
               'VectorStores' ]

BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

APP_TITLE = 'Buddy'

APP_SUBTITLE = 'Budget Execution AI'

PROMPT_ID = 'pmpt_697f53f7ddc881938d81f9b9d18d6136054cd88c36f94549'

PROMPT_VERSION = '12'

GPT_VECTORSTORE_IDS = [ 'vs_712r5W5833G6aLxIYIbuvVcK', 'vs_697f86ad98888191b967685ae558bfc0' ]

GPT_FILE_IDS = [ 'file-Wd8G8pbLSgVjHur8Qv4mdt', 'file-WPmTsHFYDLGHbyERqJdyqv', 'file-DW5TuqYoEfqFfqFFsMXBvy',
		'file-U8ExiB6aJunAeT6872HtEU', 'file-FHkNiF6Rv29eCkAWEagevT', 'file-XsjQorjtffHTWjth8EVnkL' ]

GPT_WEB_DOMAINS = [ 'congress.gov', 'google.com', 'gao.gov', 'omb.gov', 'defense.gov' ]

TEXT_TYPES = { 'output_text' }

MARKDOWN_HEADING_PATTERN = re.compile( r"^##\s+(?P<title>.+?)\s*$" )

XML_BLOCK_PATTERN  = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>", re.DOTALL )

DB_PATH = "stores/sqlite/Data.db"

ANALYST = '❓'

BUDDY = '🧠'

PROVIDERS = { 'GPT': 'gpt', 'Gemini': 'gemini', 'Groq': 'grok', }

MODE_CLASS_MAP = { 'Chat': None,
		'Text': [ 'Chat' ],
		'Images': [ 'Image' ],
		'Audio': [ 'TTS', 'Translation', 'Transcription' ],
		'Embeddings': [ 'Embedding' ],
}

CLASS_MODE_MAP = { 'GPT': cfg.GPT_MODES, 'Gemini': cfg.GEMINI_MODES, 'Grok': cfg.GROQ_MODES  }

LOGO_MAP = { 'GPT': cfg.GPT_LOGO, 'Gemini': cfg.GEMINI_LOGO, 'Groq': cfg.GROQ_LOGO }

# ==============================================================================
# UTILITIES
# ==============================================================================
def xml_converter( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def markdown_converter( markdown: str ) -> str:
	"""
		
		Purpose:
		_________
		Convert Markdown-formatted system instructions back into
		XML-delimited prompt text by treating level-2 headings (##)
		as section delimiters.
	
		Parameters:
		-----------
		markdown (str)- Markdown text using '##' headings to indicate sections.
	
		Returns:
		--------
		str - XML-delimited text suitable for storage in the prompt database.
				
	"""
	lines: List[ str ] = markdown.splitlines( )
	output: List[ str ] = [ ]
	current_tag: Optional[ str ] = None
	buffer: List[ str ] = [ ]
	
	def flush( ) -> None:
		"""
		
			Purpose:
			_________
			Emit the currently buffered section as an XML-delimited block.
			
		"""
		nonlocal current_tag, buffer
		if current_tag is None:
			return
		body: str = "\n".join( buffer ).strip( )
		output.append( f"<{current_tag}>" )
		if body:
			output.append( body )
		output.append( f"</{current_tag}>" )
		output.append( "" )
		buffer.clear( )
		for line in lines:
			match = MARKDOWN_HEADING_PATTERN.match( line )
			if match:
				flush( )
				title: str = match.group( 'title' )
				current_tag = (title.strip( ).lower( )
				               .replace( ' ', '_' ).replace( '-', '_' ))
			else:
				if current_tag is not None:
					buffer.append( line )
		flush( )
		
		# Remove trailing whitespace blocks
		while output and not output[ -1 ].strip( ):
			output.pop( )
		return "\n".join( output )

def encode_image_base64( path: str ) -> str:
	"""
	
		Purpose:
		_________
		
		Parametes:
		----------
		
		
		Returns:
		--------
		
		
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def chunk_text( text: str, size: int=1200, overlap: int=200 ) -> List[ str ]:
	chunks, i = [ ], 0
	while i < len( text ):
		chunks.append( text[ i:i + size ] )
		i += size - overlap
	return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""
	
		Purpose:
		_________
		
		
	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	
def init_state( ) -> None:
	"""
	
		Purpose:
		_________
		Initializes all session state variables.
		
		
	"""
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = [ ]
	
	if 'last_answer' not in st.session_state:
		st.session_state.last_answer = ''
	
	if 'last_sources' not in st.session_state:
		st.session_state.last_sources = [ ]
	
	if 'last_analysis' not in st.session_state:
		st.session_state.last_analysis = {
				'tables': [ ],
				'files': [ ],
				'text': [ ],
		}
	
	if 'execution_mode' not in st.session_state:
		st.session_state.execution_mode = 'Standard'

def reset_state( ) -> None:
	"""
	
		Purpose:
		_________
		Resets the session state to default values
		
	"""
	st.session_state.chat_history = [ ]
	st.session_state.last_answer = ""
	st.session_state.last_sources = [ ]
	st.session_state.last_analysis = {
			'tables': [ ],
			'files': [ ],
			'text': [ ],
	}

def normalize( obj: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		_________
		Normalizes models
		
		Parmeters:
		----------
		obj : Any
		
	"""
	if isinstance( obj, dict ):
		return obj
	if hasattr( obj, 'model_dump' ):
		return obj.model_dump( )
	return {
			k: getattr( obj, k )
			for k in dir( obj )
			if not k.startswith( "_" ) and not callable( getattr( obj, k ) )
	}

def extract_answer( response ) -> str:
	"""
	
		Purpose:
		_________
		Parses-out answer text from a response
		
		Parameters:
		------------
		response: str
		
	"""
	texts: List[ str ] = [ ]
	if not response or not getattr( response, 'output', None ):
		return ''
	
	for item in response.output:
		item_type = getattr( item, 'type', None )
		
		if item_type in TEXT_TYPES:
			text = getattr( item, 'text', None )
			if text:
				texts.append( text )
			continue
		
		content = getattr( item, 'content', None )
		if not content:
			continue
		
		for block in content:
			if getattr( block, 'type', None ) in TEXT_TYPES:
				text = getattr( block, 'text', None )
				if text:
					texts.append( text )
	
	return '\n'.join( texts ).strip( )

def extract_sources( response ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		_________
		Parses-out sources from response text.
		
	"""
	sources: List[ Dict[ str, Any ] ] = [ ]
	if not response or not getattr( response, 'output', None ):
		return sources
	
	for item in response.output:
		t = getattr( item, 'type', None )
		if t == 'web_search_call':
			raw = getattr( item.action, 'sources', None )
			if raw:
				for src in raw:
					s = normalize( src )
					sources.append( {
							'title': s.get( 'title' ),
							'snippet': s.get( 'snippet' ),
							'url': s.get( 'url' ),
							'file_id': None,
					} )
		
		# -------------------------
		# File search (vector store)
		# -------------------------
		elif t == 'file_search_call':
			raw = getattr( item, 'results', None )
			if raw:
				for r in raw:
					s = normalize( r )
					sources.append( {
							'title': s.get( 'file_name' ) or s.get( 'title' ),
							'snippet': s.get( 'text' ),
							'url': None,
							'file_id': s.get( 'file_id' ),
					} )
	
	return sources

def extract_analysis( response ) -> Dict[ str, Any ]:
	artifacts = {
			'tables': [ ],
			'files': [ ],
			'text': [ ] }
	
	if not response or not getattr( response, 'output', None ):
		return artifacts
	
	for item in response.output:
		if getattr( item, 'type', None ) != 'code_interpreter_call':
			continue
		
		outputs = getattr( item, 'outputs', None ) or [ ]
		for out in outputs:
			out_type = getattr( out, 'type', None )
			
			if out_type == 'table':
				artifacts[ 'tables' ].append( normalize( out ) )
			
			elif out_type == 'file':
				artifacts[ 'files' ].append( normalize( out ) )
			
			elif out_type in TEXT_TYPES:
				text = getattr( out, 'text', None )
				if text:
					artifacts[ 'text' ].append( text )
	
	return artifacts

def save_temp( upload ) -> str:
	"""
	
		Purpose:
		_________
		Save uploaded file to a named temporary file and return path.
		
	"""
	with tempfile.NamedTemporaryFile( delete=False ) as tmp:
		tmp.write( upload.read( ) )
		return tmp.name

def _extract_usage_from_response( resp: Any ) -> Dict[ str, int ]:
	"""
	
		Purpose:
		_________
		Extract token usage from a response object/dict.
		Returns dict with prompt_tokens, completion_tokens, total_tokens.
		Defensive: returns zeros if not present.
		
	"""
	usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0,
	}
	if not resp:
		return usage
	
	raw = None
	try:
		raw = getattr( resp, "usage", None )
	except Exception:
		raw = None
	
	if not raw and isinstance( resp, dict ):
		raw = resp.get( "usage" )
	
	if not raw:
		return usage
	
	try:
		if isinstance( raw, dict ):
			usage[ "prompt_tokens" ] = int( raw.get( "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				raw.get(
					"total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ],
				)
			)
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int(
				getattr( raw, "total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ], ) )
	except Exception:
		usage[ "total_tokens" ] = (usage[ "prompt_tokens" ] + usage[ "completion_tokens" ])
	
	return usage

def _update_token_counters( resp: Any ) -> None:
	"""
	
		Purpose:
		_________
		Update session_state.last_call_usage and accumulate into session_state.token_usage.
		
	"""
	usage = _extract_usage_from_response( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ "prompt_tokens" ] += usage.get( "prompt_tokens", 0 )
	st.session_state.token_usage[ "completion_tokens" ] += usage.get( "completion_tokens", 0 )
	st.session_state.token_usage[ "total_tokens" ] += usage.get( "total_tokens", 0 )

def _display_value( val: Any ) -> str:
	"""
		Render a friendly display string for header values.
		None -> em dash; otherwise str(value).
	"""
	if val is None:
		return "—"
	try:
		return str( val )
	except Exception:
		return "—"

def build_intent_prefix( mode: str ) -> str:
	if mode == 'Guidance Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using authoritative policy and guidance only. '
				'Do not perform financial computation.\n\n'
		)
	if mode == 'Analysis Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using financial analysis and computation only. '
				'Minimize policy citation.\n\n'
		)
	return ''

def ensure_db( ) -> None:
	Path( "stores/sqlite" ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS chat_history
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          role
                          TEXT,
                          content
                          TEXT
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          chunk
                          TEXT,
                          vector
                          BLOB
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS Prompts
                      (
                          PromptsId
                          INTEGER
                          NOT
                          NULL
                          UNIQUE,
                          Name
                          TEXT
                      (
                          80
                      ),
                          Text TEXT,
                          Version TEXT
                      (
                          80
                      ),
                          ID TEXT
                      (
                          80
                      ),
                          PRIMARY KEY
                      (
                          PromptsId
                          AUTOINCREMENT
                      )
                          )
		              """ )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( "SELECT role, content FROM chat_history ORDER BY id" ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn
		)
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE Name=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"INSERT INTO Prompts (Name, Text, Version, ID) VALUES (?, ?, ?, ?)",
			(data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ])
		)

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ], pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"
	
	if st.session_state.use_semantic:
		with sqlite3.connect( DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_sim( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in st.session_state.basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	for r, c in st.session_state.messages:
		prompt += f"<|{r}|>\n{c}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

# ==============================================================================
# Page Setup / Configuration
# ==============================================================================
client = OpenAI( )

AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BUDDY, }

st.set_page_config( page_title=cfg.APP_TITLE, layout='wide',
	page_icon=cfg.FAVICON, initial_sidebar_state='collapsed' )

st.caption( cfg.APP_SUBTITLE )

inject_response_css( )

init_state( )

# ======================================================================================
# Session State — initialize per-mode model keys and token counters
# ======================================================================================
if 'openai_api_key' not in st.session_state:
	st.session_state.openai_api_key = ''

if 'gemini_api_key' not in st.session_state:
	st.session_state.gemini_api_key = ''

if 'groq_api_key' not in st.session_state:
	st.session_state.groq_api_key = ''

if 'google_api_key' not in st.session_state:
	st.session_state.google_api_key = ''

if 'xai_api_key' not in st.session_state:
	st.session_state.xai_api_key = ''


if st.session_state.openai_api_key == '':
	default = cfg.OPENAI_API_KEY
	if default:
		st.session_state.openai_api_key = default
		os.environ[ 'OPENAI_API_KEY' ] = default

if st.session_state.gemini_api_key == '':
	default = getattr( cfg, 'GEMINI_API_KEY', '' )
	if default:
		st.session_state.gemini_api_key = default
		os.environ[ 'GEMINI_API_KEY' ] = default

if st.session_state.groq_api_key == '':
	default = getattr( cfg, 'GROQ_API_KEY', '' )
	if default:
		st.session_state.groq_api_key = default
		os.environ[ 'GROQ_API_KEY' ] = default

if st.session_state.google_api_key == '':
	default = getattr( cfg, 'GOOGLE_API_KEY', '' )
	if default:
		st.session_state.google_api_key = default
		os.environ[ 'GOOGLE_API_KEY' ] = default

if st.session_state.xai_api_key == '':
	default = getattr( cfg, 'XAI_API_KEY', '' )
	if default:
		st.session_state.xai_api_key = default
		os.environ[ 'XAI_API_KEY' ] = default

if 'provider' not in st.session_state or st.session_state[ 'provider' ] is None:
	st.session_state[ 'provider' ] = 'GPT'

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Text'

if 'messages' not in st.session_state:
	st.session_state.messages: List[ Dict[ str, Any ] ] = [ ]

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0,
	}

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0,
	}

if 'files' not in st.session_state:
	st.session_state.files: List[ str ] = [ ]

if 'text_model' not in st.session_state:
	st.session_state[ 'text_model' ] = None
	
if 'image_model' not in st.session_state:
	st.session_state[ 'image_model' ] = None
	
if 'audio_model' not in st.session_state:
	st.session_state[ 'audio_model' ] = None
	
if 'embed_model' not in st.session_state:
	st.session_state[ 'embed_model' ] = None
	
if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.7
	
if 'top_p' not in st.session_state:
	st.session_state[ 'top_p' ] = 1.0
	
if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 512
	
if 'freq_penalty' not in st.session_state:
	st.session_state[ 'freq_penalty' ] = 0.0
	
if 'pres_penalty' not in st.session_state:
	st.session_state[ 'pres_penalty' ] = 0.0
	
if 'stop_sequences' not in st.session_state:
	st.session_state[ 'stop_sequences' ] = [ ]

if 'provider' not in st.session_state:
	st.session_state[ 'provider' ] = 'GPT'

if 'api_keys' not in st.session_state:
	st.session_state.api_keys = { 'GPT': None,
			'Groq': None,
			'Gemini': None,}

# ======================================================================================
#  PROVIDER
# ======================================================================================
def get_provider_module( ):
	provider = st.session_state.get( 'provider', 'GPT' )
	module_name = cfg.PROVIDERS.get( provider, 'gpt' )
	return __import__( module_name )

def get_chat_instance( ):
	"""

		Purpose:
		-------
		Returns a Chat() instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Chat( )

def _provider( ):
	return st.session_state.get( 'provider', 'GPT' )

def _safe( module, attr, fallback ):
	try:
		mod = __import__( module )
		return getattr( mod, attr, fallback )
	except Exception:
		return fallback

# ---------------- TEXT ----------------
def text_model_options( chat ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'model_options', chat.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'model_options', chat.model_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'model_options', chat.model_options )
	if _provider( ) == 'Mistral':
		return _safe( 'missy', 'model_options', chat.model_options )
	if _provider( ) == 'Claude':
		return _safe( 'claude', 'model_options', chat.model_options )
	return chat.model_options

# ---------------- IMAGES ----------------
def image_model_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'image_model_options', image.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'image_model_options', image.model_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'image_model_options', image.model_options )
	if _provider( ) == 'Mistral':
		return _safe( 'missy', 'image_model_options', image.model_options )
	if _provider( ) == 'Claude':
		return _safe( 'claude', 'image_model_options', image.model_options )
	return image.model_options

def image_size_or_aspect_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'aspect_options', image.size_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'aspect_options', image.size_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'aspect_options', image.size_options )
	if _provider( ) == 'Mistral':
		return _safe( 'missy', 'aspect_options', image.size_options )
	if _provider( ) == 'Claude':
		return _safe( 'claude', 'aspect_options', image.size_options )
	return image.size_options

# ---------------- AUDIO ----------------
def audio_model_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'audio_model_options', transcriber.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'audio_model_options', transcriber.model_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'audio_model_options', transcriber.model_options )
	return transcriber.model_options

def audio_language_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'language_options', transcriber.language_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'language_options', transcriber.language_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'language_options', transcriber.language_options )
	return transcriber.language_options

# ---------------- EMBEDDINGS ----------------
def embedding_model_options( embed ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'embedding_model_options', embed.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'embedding_model_options', embed.model_options )
	if _provider( ) == 'Groq':
		return _safe( 'grok', 'embedding_model_options', embed.model_options )
	return embed.model_options

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
	logo_slot = st.empty( )
	provider = st.session_state.get( "provider", "GPT" )

	style_subheaders( )
	st.subheader( "" )
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	provider = st.selectbox( "API", list( PROVIDERS.keys( ) ),
		index=list( cfg.PROVIDERS.keys( ) ).index( st.session_state.get( "provider", "GPT" ) ) )
	
	st.session_state[ "provider" ] = provider
	logo_path = LOGO_MAP.get( provider )
	
	with st.expander( '🔑 Keys:', expanded=False ):
		openai_key = st.text_input(
			'OpenAI API Key',
			type='password',
			value=st.session_state.openai_api_key or '',
			help='Overrides OPENAI_API_KEY from config.py for this session only.'
		)
		
		gemini_key = st.text_input(
			'Gemini API Key',
			type='password',
			value=st.session_state.gemini_api_key or '',
			help='Overrides GEMINI_API_KEY from config.py for this session only.'
		)
		
		groq_key = st.text_input(
			'Groq API Key',
			type='password',
			value=st.session_state.groq_api_key or '',
			help='Overrides GROQ_API_KEY from config.py for this session only.'
		)
		
		google_key = st.text_input(
			'Google API Key',
			type='password',
			value=st.session_state.google_api_key or '',
			help='Overrides GOOGLE_API_KEY from config.py for this session only.'
		)
		
		xai_key = st.text_input(
			'xAi API Key',
			type='password',
			value=st.session_state.xai_api_key or '',
			help='Overrides XAI_API_KEY from config.py for this session only.'
		)
		
		if openai_key:
			st.session_state.openai_api_key = openai_key
			os.environ[ 'OPENAI_API_KEY' ] = openai_key
		
		if gemini_key:
			st.session_state.gemini_api_key = gemini_key
			os.environ[ 'GEMINI_API_KEY' ] = gemini_key
		
		if groq_key:
			st.session_state.groq_api_key = groq_key
			os.environ[ 'GROQ_API_KEY' ] = groq_key
		
		if google_key:
			st.session_state.google_api_key = google_key
			os.environ[ 'GOOGLE_API_KEY' ] = google_key
		
		if xai_key:
			st.session_state.xai_api_key = xai_key
			os.environ[ 'XAI_API_KEY' ] = xai_key
	
	with logo_slot:
		if logo_path and os.path.exists( logo_path ):
			col1, col2, col3 = st.columns( [ 1,  2,  1 ] )
			with col2:
				logo_path = LOGO_MAP.get( provider )
				if logo_path and os.path.exists( logo_path ):
					st.logo( logo_path, size='large', link=cfg.CRS )
					
	
	if st.button( 'Clear Chat' ):
		reset_state( )
		st.rerun( )
		
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	if provider == 'Gemini':
		mode = st.sidebar.radio( 'Select Mode', cfg.GEMINI_MODES, index=0 )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	elif provider == 'Grok':
		mode = st.sidebar.radio( 'Select Mode', cfg.GROQ_MODES, index=0 )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	else:
		mode = st.sidebar.radio( 'Select Mode', cfg.GPT_MODES, index=0 )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
# =============================================================================
# CHAT MODE
# =============================================================================
if mode == 'Chat':
	st.subheader( "🧠 Chat Completions" )
	st.divider( )
	st.header( '' )
	provider_module = get_provider_module( )
	
	# ------------------------------------------------------------------
	# Sidebar — Text Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.text( '⚙️  Chat Settings' )
		st.radio( 'Execution Mode',
			options=[ 'Standard', 'Guidance Only', 'Analysis Only' ],
			index=[ 'Standard', 'Guidance Only',
			        'Analysis Only' ].index( st.session_state.execution_mode ),
			key='execution_mode', )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.25,  3.5,  0.25 ] )
	
	with center:
		user_input = st.chat_input( "Do you have a Planning, Programming, or Budget Execution question?" )
		
		if user_input:
			# -------------------------------
			# Render user message
			# -------------------------------
			with st.chat_message( "user", avatar=ANALYST ):
				st.markdown( user_input )
			
			# -------------------------------
			# Run prompt
			# -------------------------------
			with st.chat_message( 'assistant', avatar=BUDDY ):
				try:
					with st.spinner( 'Running prompt...' ):
						response = client.responses.create(
							prompt={ 'id': PROMPT_ID, 'version': PROMPT_VERSION, },
							input=[ { 'role': 'user', 'content': [ { 'type': 'input_text',
							                                         'text': user_input, } ], } ],
							tools=[ { 'type': 'file_search', 'vector_store_ids': VECTOR_STORE_IDS, },
									{ 'type': 'web_search', 'filters': { 'allowed_domains': WEB_DOMAINS, },
											'search_context_size': 'medium',
											'user_location': { 'type': 'approximate' },
									},
									{ 'type': 'code_interpreter',
									  'container': { 'type': 'auto', 'file_ids': FILE_IDS, },
									},
							],
							include=[
									'web_search_call.action.sources',
									'code_interpreter_call.outputs',
							],
							store=True,
						)
					sources = st.session_state.get( "last_sources", [ ] )
		
					if sources:
						st.markdown( "#### Sources" )
						for i, src in enumerate( sources, 1 ):
							url = src.get( "url" )
							title = src.get( "title" ) or src.get( "file_name" ) or f"Source {i}"
							
							if url:
								st.markdown( f"- [{title}]({url})" )
							elif src.get( "file_id" ):
								st.markdown( f"- {title} _(Vector Store File: `{src[ 'file_id' ]}`)_" )
					
					# -------------------------------
					# Extract and render text output
					# -------------------------------
					output_text = ""
				
					for item in response.output:
						if item.type == "message":
							for part in item.content:
								if part.type == "output_text":
									output_text += part.text
					
					if output_text.strip( ):
						st.markdown( output_text )
					else:
						st.warning( 'No text response returned by the prompt.' )
					
					# -------------------------------
					# Persist minimal chat history
					# -------------------------------
					st.session_state.chat_history.append(
						{
							'role': 'user',
							'content': user_input
						}
					)
					st.session_state.chat_history.append(
						{
								'role': 'assistant',
								'content': output_text
						}
					)
				except Exception as e:
					st.error( "An error occurred while running the prompt." )
					st.exception( e )

# ======================================================================================
# TEXT MODE
# ======================================================================================
elif mode == "Text":
	st.subheader( "📝 Text Generation" )
	st.divider( )
	st.header( '' )
	provider_module = get_provider_module( )
	chat = provider_module.Chat( )
	
	# ------------------------------------------------------------------
	# Sidebar — Text Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.text( '⚙️ Text Settings' )
		
		# ---------------- Model ----------------
		text_model = st.selectbox( 'Model', chat.model_options,
			index=(chat.model_options.index( st.session_state[ 'text_model' ] )
			       if st.session_state.get( 'text_model' ) in chat.model_options
			       else 0), )
		st.session_state[ 'text_model' ] = text_model
		
		# ---------------- Parameters ----------------
		with st.expander( '🔣 Parameters:', expanded=True ):
			temperature = st.slider( 'Temperature', min_value=0.0, max_value=1.0,
				value=float( st.session_state.get( 'temperature', 0.7 ) ), step=0.01, )
			st.session_state[ 'temperature' ] = float( temperature )
			
			top_p = st.slider(
				'Top-P',
				min_value=0.0,
				max_value=1.0,
				value=float( st.session_state.get( 'top_p', 1.0 ) ),
				step=0.01,
			)
			st.session_state[ 'top_p' ] = float( top_p )
			
			max_tokens = st.number_input(
				'Max Tokens',
				min_value=1,
				max_value=100000,
				value=int( st.session_state.get( 'max_tokens', 512 ) ),
			)
			st.session_state[ 'max_tokens' ] = int( max_tokens )
			
			freq_penalty = st.slider(
				'Frequency Penalty',
				min_value=-2.0,
				max_value=2.0,
				value=float( st.session_state.get( 'freq_penalty', 0.0 ) ),
				step=0.01,
			)
			st.session_state[ 'freq_penalty' ] = float( freq_penalty )
			
			pres_penalty = st.slider(
				'Presence Penalty',
				min_value=-2.0,
				max_value=2.0,
				value=float( st.session_state.get( 'pres_penalty', 0.0 ) ),
				step=0.01,
			)
			st.session_state[ 'pres_penalty' ] = float( pres_penalty )
			
			stop_text = st.text_area(
				'Stop Sequences (one per line)',
				value='\n'.join( st.session_state.get( 'stop_sequences', [ ] ) ),
				height=80,
			)
			st.session_state[ 'stop_sequences' ] = [
					s for s in stop_text.splitlines( ) if s.strip( )
			]
		
		# ---------------- Include options ----------------
		if mode == 'GPT':
			include = st.multiselect( 'Include:', chat.include_options )
			chat.include = include
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	
	with center:
		for msg in st.session_state.messages:
			with st.chat_message( msg[ 'role' ] ):
				st.markdown( msg[ 'content' ] )
		
		prompt = st.chat_input( 'Ask Boo…' )
		if prompt is not None:
			st.session_state.messages.append( {
					'role': 'user',
					'content': prompt } )
			
			with st.chat_message( 'assistant' ):
				gen_kwargs = { }
			
			with st.spinner( 'Thinking…' ):
				gen_kwargs[ 'model' ] = st.session_state[ 'text_model' ]
				gen_kwargs[ 'top_p' ] = st.session_state[ 'top_p' ]
				gen_kwargs[ 'max_tokens' ] = st.session_state[ 'max_tokens' ]
				gen_kwargs[ 'frequency' ] = st.session_state[ 'freq_penalty' ]
				gen_kwargs[ 'presence' ] = st.session_state[ 'pres_penalty' ]
			
				if st.session_state[ 'stop_sequences' ]:
					gen_kwargs[ 'stops' ] = st.session_state[ 'stop_sequences' ]
				
				response = None
				try:
					mdl = str( gen_kwargs[ 'model' ] )
					if mdl.startswith( 'gpt-5' ):
						response = chat.generate_text( prompt=prompt, model=gen_kwargs[ 'model' ] )
					else:
						response = chat.generate_text( )
				except Exception as exc:
					err = Error( exc )
					st.error( f'Generation Failed: {err.info}' )
					response = None
				
				st.markdown( response )
				st.session_state.messages.append( {
						'role': 'assistant',
						'content': response } )
				
				try:
					_update_token_counters(
						getattr( chat, 'response', None ) or response
					)
				except Exception:
					pass
	
	lcu = st.session_state.last_call_usage
	tu = st.session_state.token_usage
	
	if any( lcu.values( ) ):
		st.info(
			f"Last call — prompt: {lcu[ 'prompt_tokens' ]}, "
			f"completion: {lcu[ 'completion_tokens' ]}, "
			f"total: {lcu[ 'total_tokens' ]}"
		)
	
	if tu[ "total_tokens" ] > 0:
		st.write(
			f"Session totals — prompt: {tu[ 'prompt_tokens' ]} · "
			f"completion: {tu[ 'completion_tokens' ]} · "
			f"total: {tu[ 'total_tokens' ]}"
		)

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == "Images":
	st.subheader( '📷 Image API')
	provider_module = get_provider_module( )
	image = provider_module.Images( )
	
	# ------------------------------------------------------------------
	# Sidebar — Image Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.text( '⚙️ Image Settings' )
		
		# ---------------- Model ----------------
		image_model = st.selectbox( "Model", image.model_options,
			index=( image.model_options.index( st.session_state[ "image_model" ] )
					if st.session_state.get( "image_model" ) in image.model_options
					else 0 ),
		)
		st.session_state[ "image_model" ] = image_model
		
		# ---------------- Size / Aspect Ratio (provider-aware) ----------------
		if hasattr( image, "aspect_options" ):
			size_or_aspect = st.selectbox( "Aspect Ratio", image.aspect_options, )
			size_arg = size_or_aspect
		else:
			size_or_aspect = st.selectbox( "Size", image.size_options, )
			size_arg = size_or_aspect
		
		# ---------------- Quality ----------------
		quality = None
		if hasattr( image, 'quality_options' ):
			quality = st.selectbox( 'Quality', image.quality_options, )
		
		# ---------------- Format ----------------
		fmt = None
		if hasattr( image, 'format_options' ):
			fmt = st.selectbox( 'Format', image.format_options, )
	
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		# ------------------------------------------------------------------
		# Main UI — Tabs
		# ------------------------------------------------------------------
		tab_gen, tab_analyze = st.tabs( [ 'Generate', 'Analyze' ] )
		with tab_gen:
			prompt = st.text_area( 'Prompt' )
			if st.button( 'Generate Image' ):
				with st.spinner( 'Generating…' ):
					try:
						kwargs: Dict[ str, Any ] = {
								'prompt': prompt,
								'model': image_model,
						}
						
						# Provider-safe optional args
						if size_arg is not None:
							kwargs[ 'size' ] = size_arg
						if quality is not None:
							kwargs[ 'quality' ] = quality
						if fmt is not None:
							kwargs[ 'fmt' ] = fmt
						
						img_url = image.generate( **kwargs )
						st.image( img_url )
						
						try:
							_update_token_counters(
								getattr( image, 'response', None )
							)
						except Exception:
							pass
					
					except Exception as exc:
						st.error( f'Image generation failed: {exc}' )
		
		with tab_analyze:
			st.markdown( 'Image analysis — upload an image to analyze.' )
			
			uploaded_img = st.file_uploader(
				'Upload an image for analysis',
				type=[ 'png',
				       'jpg',
				       'jpeg',
				       'webp' ],
				accept_multiple_files=False,
				key='images_analyze_uploader',
			)
			
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				
				st.image(
					uploaded_img,
					caption='Uploaded image preview',
					use_column_width=True,
				)
				
				# Discover available analysis methods on Image object
				available_methods = [ ]
				for candidate in (
							'analyze',
							'describe_image',
							'describe',
							'classify',
							'detect_objects',
							'caption',
							'image_analysis',
				):
					if hasattr( image, candidate ):
						available_methods.append( candidate )
				
				if available_methods:
					chosen_method = st.selectbox(
						'Method',
						available_methods,
						index=0,
					)
				else:
					chosen_method = None
					st.info(
						'No dedicated image analysis method found on Image object; '
						'attempting generic handlers.'
					)
				
				chosen_model = st.selectbox(
					"Model (analysis)",
					[ image_model,
					  None ],
					index=0,
				)
				
				chosen_model_arg = (
						image_model if chosen_model is None else chosen_model
				)
				
				if st.button( "Analyze Image" ):
					with st.spinner( "Analyzing image…" ):
						analysis_result = None
						try:
							if chosen_method:
								func = getattr( image, chosen_method, None )
								if func:
									try:
										analysis_result = func( tmp_path )
									except TypeError:
										analysis_result = func(
											tmp_path, model=chosen_model_arg
										)
							else:
								for fallback in (
											"analyze",
											"describe_image",
											"describe",
											"caption",
								):
									if hasattr( image, fallback ):
										func = getattr( image, fallback )
										try:
											analysis_result = func( tmp_path )
											break
										except Exception:
											continue
							
							if analysis_result is None:
								st.warning(
									"No analysis output returned by the available methods."
								)
							else:
								if isinstance( analysis_result, (dict, list) ):
									st.json( analysis_result )
								else:
									st.markdown( "**Analysis result:**" )
									st.write( analysis_result )
								
								try:
									_update_token_counters(
										getattr( image, "response", None )
										or analysis_result
									)
								except Exception:
									pass
						
						except Exception as exc:
							st.error( f"Analysis Failed: {exc}" )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == "Audio":
	# ------------------------------------------------------------------
	# Provider-aware Audio instantiation
	# ------------------------------------------------------------------
	provider_module = get_provider_module( )
	st.subheader( '🔉 Audio API')
	st.divider( )
	transcriber = None
	translator = None
	tts = None
	
	if hasattr( provider_module, "Transcription" ):
		transcriber = provider_module.Transcription( )
	if hasattr( provider_module, "Translation" ):
		translator = provider_module.Translation( )
	if hasattr( provider_module, "TTS" ):
		tts = provider_module.TTS( )
	
	# ------------------------------------------------------------------
	# Sidebar — Audio Settings (NO functionality removed)
	# ------------------------------------------------------------------
	with st.sidebar:
		st.text( '⚙️ Audio Settings' )
		
		# ---------------- Task ----------------
		available_tasks = [ ]
		if transcriber is not None:
			available_tasks.append( "Transcribe" )
		if translator is not None:
			available_tasks.append( "Translate" )
		if tts is not None:
			available_tasks.append( "Text-to-Speech" )
		
		if not available_tasks:
			st.info( "Audio is not supported by the selected provider." )
			task = None
		else:
			task = st.selectbox( "Task", available_tasks )
		
		# ---------------- Model (provider-correct) ----------------
		audio_model = None
		model_options = [ ]
		
		if task == "Transcribe" and transcriber and hasattr( transcriber, "model_options" ):
			model_options = transcriber.model_options
		elif task == "Translate" and translator and hasattr( translator, "model_options" ):
			model_options = translator.model_options
		elif task == "Text-to-Speech" and tts and hasattr( tts, "model_options" ):
			model_options = tts.model_options
		
		if model_options:
			audio_model = st.selectbox(
				"Model",
				model_options,
				index=(
						model_options.index( st.session_state.get( "audio_model" ) )
						if st.session_state.get( "audio_model" ) in model_options
						else 0
				),
			)
			st.session_state[ "audio_model" ] = audio_model
		
		# ---------------- Language / Voice Options ----------------
		language = None
		voice = None
		
		if task in ("Transcribe", "Translate"):
			obj = transcriber if task == "Transcribe" else translator
			if obj and hasattr( obj, "language_options" ):
				language = st.selectbox(
					"Language",
					obj.language_options,
				)
		
		if task == "Text-to-Speech" and tts:
			if hasattr( tts, "voice_options" ):
				voice = st.selectbox(
					"Voice",
					tts.voice_options,
				)
	
	# ------------------------------------------------------------------
	# Main UI — Audio Input / Output
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.25,  3.5, 0.25 ] )
	
	with center:
		if task in ('Transcribe', 'Translate'):
			uploaded = st.file_uploader(
				'Upload audio file',
				type=[ 'wav',
				       'mp3',
				       'm4a',
				       'flac' ],
			)
			
			if uploaded:
				tmp_path = save_temp( uploaded )
				
				if task == 'Transcribe' and transcriber:
					with st.spinner( 'Transcribing…' ):
						try:
							text = transcriber.transcribe(
								tmp_path,
								model=audio_model,
								language=language,
							)
							st.text_area( 'Transcript', value=text, height=300 )
							
							try:
								_update_token_counters(
									getattr( transcriber, 'response', None )
								)
							except Exception:
								pass
						
						except Exception as exc:
							st.error( f'Transcription failed: {exc}' )
				
				elif task == 'Translate' and translator:
					with st.spinner( 'Translating…' ):
						try:
							text = translator.translate(
								tmp_path,
								model=audio_model,
								language=language,
							)
							st.text_area( 'Translation', value=text, height=300 )
							
							try:
								_update_token_counters(
									getattr( translator, 'response', None )
								)
							except Exception:
								pass
						
						except Exception as exc:
							st.error( f'Translation failed: {exc}' )
		
		elif task == 'Text-to-Speech' and tts:
			text = st.text_area( 'Text to synthesize' )
			
			if text and st.button( 'Generate Audio' ):
				with st.spinner( 'Synthesizing speech…' ):
					try:
						audio_bytes = tts.speak(
							text,
							model=audio_model,
							voice=voice,
						)
						st.audio( audio_bytes )
						
						try:
							_update_token_counters(
								getattr( tts, "response", None )
							)
						except Exception:
							pass
					
					except Exception as exc:
						st.error( f"Text-to-speech failed: {exc}" )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	provider_module = get_provider_module( )
	st.subheader( '⛓️  Vector Embeddings')
	st.divider( )
	if not hasattr( provider_module, 'Embeddings' ):
		st.info( 'Embeddings are not supported by the selected provider.' )
	else:
		embed = provider_module.Embeddings( )
		with st.sidebar:
			st.text( '⚙️ Embedding Settings' )
			
			embed_model = st.selectbox( 'Model', embed.model_options,
				index=( embed.model_options.index( st.session_state[ 'embed_model' ] )
						if st.session_state.get( 'embed_model' ) in embed.model_options
						else 0 ),
			)
			st.session_state[ 'embed_model' ] = embed_model
			method = None
			if hasattr( embed, "methods" ):
				method = st.selectbox( "Method", embed.methods, )
		
		# ------------------------------------------------------------------
		# Main UI — Embedding execution (unchanged behavior)
		# ------------------------------------------------------------------
		left, center, right = st.columns( [ 0.25,  3.5,  0.25 ] )
		with center:
			text = st.text_area( 'Text to embed' )
			
			if text and st.button( 'Embed' ):
				with st.spinner( 'Embedding…' ):
					try:
						if method:
							vector = embed.create(
								text,
								model=embed_model,
								method=method,
							)
						else:
							vector = embed.create( text, model=embed_model, )
						
						st.write( 'Vector length:', len( vector ) )
						
						try:
							_update_token_counters(
								getattr( embed, 'response', None )
							)
						except Exception:
							pass
					
					except Exception as exc:
						st.error( f'Embedding failed: {exc}' )

# ======================================================================================
# VECTOR MODE
# ======================================================================================
elif mode == "Vector Store":
	try:
		chat  # type: ignore
	except NameError:
		chat = get_provider_module( ).Chat( )
	
	st.subheader( '🕸️ Vector Stores')
	st.divider( )
	vs_map = getattr( chat, "vector_stores", None )
	if vs_map and isinstance( vs_map, dict ):
		st.markdown( "**Known vector stores (local mapping)**" )
		for name, vid in vs_map.items( ):
			st.write( f"- **{name}** — `{vid}`" )
		st.markdown( "---" )
	
	with st.expander( "Create Vector Store", expanded=False ):
		new_store_name = st.text_input( "New store name" )
		if st.button( "Create store" ):
			if not new_store_name:
				st.warning( "Enter a store name." )
			else:
				try:
					if hasattr( chat, "create_store" ):
						res = chat.create_store( new_store_name )
						st.success( f"Create call submitted for '{new_store_name}'." )
					else:
						st.warning( "create_store method not found on chat object." )
				except Exception as exc:
					st.error( f"Create store failed: {exc}" )
	
	st.markdown( "**Manage Stores**" )
	options: List[ tuple ] = [ ]
	if vs_map and isinstance( vs_map, dict ):
		options = list( vs_map.items( ) )
	
	if not options:
		try:
			client = getattr( chat, 'client', None )
			if (
					client
					and hasattr( client, 'vector_stores' )
					and hasattr( client.vector_stores, 'list' )
			):
				api_list = client.vector_stores.list( )
				temp: List[ tuple ] = [ ]
				for item in getattr( api_list, 'data', [ ] ) or api_list:
					nm = getattr( item, 'name', None ) or (
							item.get( 'name' )
							if isinstance( item, dict )
							else None
					)
					vid = getattr( item, 'id', None ) or (
							item.get( 'id' )
							if isinstance( item, dict )
							else None
					)
					if nm and vid:
						temp.append( (nm, vid) )
				if temp:
					options = temp
		except Exception:
			options = [ ]
	
	if options:
		names = [ f"{n} — {i}" for n, i in options ]
		sel = st.selectbox( "Select a vector store", options=names )
		sel_id: Optional[ str ] = None
		sel_name: Optional[ str ] = None
		for n, i in options:
			label = f"{n} — {i}"
			if label == sel:
				sel_id = i
				sel_name = n
				break
		
		c1, c2 = st.columns( [ 1, 1 ] )
		with c1:
			if st.button( "Retrieve store" ):
				try:
					if sel_id and hasattr( chat, "retrieve_store" ):
						vs = chat.retrieve_store( sel_id )
						st.json(
							vs.__dict__
							if hasattr( vs, "__dict__" )
							else vs
						)
					else:
						st.warning('retrieve_store not available on chat object or no store selected.' )
				except Exception as exc:
					st.error( f'Retrieve failed: {exc}' )
		
		with c2:
			if st.button( 'Delete store' ):
				try:
					if sel_id and hasattr( chat, 'delete_store' ):
						res = chat.delete_store( sel_id )
						st.success( f'Delete returned: {res}' )
					else:
						st.warning(
							'delete_store not available on chat object '
							'or no store selected.'
						)
				except Exception as exc:
					st.error( f"Delete failed: {exc}" )
	else:
		st.info(
			"No vector stores discovered. Create one or confirm "
			"`chat.vector_stores` mapping exists." )

# ======================================================================================
# DOCUMENTS MODE
# ======================================================================================
elif mode == 'Documents':
	st.subheader( '📚 Document Q & A')
	st.divider( )
	left, center, right = st.columns( [ 0.25,  3.5, 0.25 ] )
	with center:
		uploaded = st.file_uploader( 'Upload documents (session only)',
			type=[ 'pdf', 'txt', 'md', 'docx' ], accept_multiple_files=True, )
		
		if uploaded:
			for up in uploaded:
				st.session_state.files.append( save_temp( up ) )
			st.success( f"Saved {len( uploaded )} file(s) to session" )
		
		if st.session_state.files:
			st.markdown( "**Uploaded documents (session-only)**" )
			idx = st.selectbox( "Choose a document",
				options=list( range( len( st.session_state.files ) ) ),
				format_func=lambda i: st.session_state.files[ i ], )
			selected_path = st.session_state.files[ idx ]
			
			c1, c2 = st.columns( [ 1, 1 ] )
			with c1:
				if st.button( "Remove selected document" ):
					removed = st.session_state.files.pop( idx )
					st.success( f"Removed {removed}" )
			with c2:
				if st.button( "Show selected path" ):
					st.info( f"Local temp path: {selected_path}" )
			
			st.markdown( "---" )
			question = st.text_area(
				"Ask a question about the selected document"
			)
			if st.button( "Ask Document" ):
				if not question:
					st.warning( "Enter a question before asking." )
				else:
					with st.spinner( "Running document Q&A…" ):
						try:
							try:
								chat  # type: ignore
							except NameError:
								chat = get_provider_module( ).Chat( )
							answer = None
							if hasattr( chat, "summarize_document" ):
								try:
									answer = chat.summarize_document(
										prompt=question,
										pdf_path=selected_path,
									)
								except TypeError:
									answer = chat.summarize_document(
										question, selected_path
									)
							elif hasattr( chat, "ask_document" ):
								answer = chat.ask_document(
									selected_path, question
								)
							elif hasattr( chat, "document_qa" ):
								answer = chat.document_qa(
									selected_path, question
								)
							else:
								raise RuntimeError(
									"No document-QA method found on chat object."
								)
							
							st.markdown( "**Answer:**" )
							st.markdown( answer or "No answer returned." )
							
							st.session_state.messages.append(
								{
										"role": "user",
										"content": f"[Document question] {question}",
								}
							)
							st.session_state.messages.append(
								{
										"role": "assistant",
										"content": answer or "",
								}
							)
							
							try:
								_update_token_counters(
									getattr( chat, "response", None )
									or answer
								)
							except Exception:
								pass
						except Exception as e:
							st.error(
								f"Document Q&A failed: {e}"
							)
		else:
			st.info(
				"No client-side documents uploaded this session. "
				"Use the uploader in the sidebar to add files."
			)

# ======================================================================================
# FILES API MODE
# ======================================================================================
elif mode == "Files":
	try:
		chat  # type: ignore
	except NameError:
		chat = Chat( )
	
	st.subheader( '📁 Files API' )
	st.divider( )
	left, center, right = st.columns( [ 0.25,  3.5,  0.25 ] )
	with center:
		list_method = None
		for name in (
					'retrieve_files',
					'retreive_files',
					'list_files',
					'get_files',
		):
			if hasattr( chat, name ):
				list_method = getattr( chat, name )
				break
		
		uploaded_file = st.file_uploader(
			'Upload file (server-side via Files API)',
			type=[
					'pdf',
					'txt',
					'md',
					'docx',
					'png',
					'jpg',
					'jpeg',
			],
		)
		if uploaded_file:
			tmp_path = save_temp( uploaded_file )
			upload_fn = None
			for name in ("upload_file", "upload", "files_upload"):
				if hasattr( chat, name ):
					upload_fn = getattr( chat, name )
					break
			if not upload_fn:
				st.warning(
					"No upload function found on chat object (upload_file)."
				)
			else:
				with st.spinner( "Uploading to Files API..." ):
					try:
						fid = upload_fn( tmp_path )
						st.success( f"Uploaded; file id: {fid}" )
					except Exception as exc:
						st.error( f"Upload failed: {exc}" )
		
		if st.button( "List files" ):
			if not list_method:
				st.warning(
					"No file-listing method found on chat object."
				)
			else:
				with st.spinner( "Listing files..." ):
					try:
						files_resp = list_method( )
						files_list = [ ]
						if files_resp is None:
							files_list = [ ]
						elif isinstance( files_resp, dict ):
							files_list = (
									files_resp.get( "data" )
									or files_resp.get( "files" )
									or [ ]
							)
						elif isinstance( files_resp, list ):
							files_list = files_resp
						else:
							try:
								files_list = getattr(
									files_resp, "data", files_resp
								)
							except Exception:
								files_list = [ files_resp ]
						
						rows = [ ]
						for f in files_list:
							try:
								fid = (
										f.get( "id" )
										if isinstance( f, dict )
										else getattr( f, "id", None )
								)
								name = (
										f.get( "filename" )
										if isinstance( f, dict )
										else getattr(
											f, "filename", None
										)
								)
								purpose = (
										f.get( "purpose" )
										if isinstance( f, dict )
										else getattr(
											f, "purpose", None
										)
								)
							except Exception:
								fid = None
								name = str( f )
								purpose = None
							rows.append(
								{
										"id": fid,
										"filename": name,
										"purpose": purpose,
								}
							)
						if rows:
							st.table( rows )
						else:
							st.info( "No files returned." )
					except Exception as exc:
						st.error( f"List files failed: {exc}" )
		
		if "files_list" in locals( ) and files_list:
			file_ids = [
					r.get( "id" )
					if isinstance( r, dict )
					else getattr( r, "id", None )
					for r in files_list
			]
			sel = st.selectbox(
				"Select file id to delete", options=file_ids
			)
			if st.button( "Delete selected file" ):
				del_fn = None
				for name in ("delete_file", "delete", "files_delete"):
					if hasattr( chat, name ):
						del_fn = getattr( chat, name )
						break
				if not del_fn:
					st.warning(
						"No delete function found on chat object."
					)
				else:
					with st.spinner( "Deleting file..." ):
						try:
							res = del_fn( sel )
							st.success( f"Delete result: {res}" )
						except Exception as exc:
							st.error( f"Delete failed: {exc}" )

# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == "Prompt Engineering":
	import sqlite3
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	
	st.subheader( '📝 Prompt Engineering' )
	st.divider( )
	# ------------------------------------------------------------------
	# Session state (single source of truth)
	# ------------------------------------------------------------------
	st.session_state.setdefault( 'pe_page', 1 )
	st.session_state.setdefault( 'pe_search', "" )
	st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
	st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
	st.session_state.setdefault( 'pe_selected_id', None )
	
	st.session_state.setdefault( 'pe_name', "" )
	st.session_state.setdefault( 'pe_text', "" )
	st.session_state.setdefault( 'pe_version', 1 )
	
	# ------------------------------------------------------------------
	# DB helpers
	# ------------------------------------------------------------------
	def get_conn( ):
		return sqlite3.connect( cfg.DB_PATH )
	
	def reset_selection( ):
		st.session_state.pe_selected_id = None
		st.session_state.pe_name = ""
		st.session_state.pe_text = ""
		st.session_state.pe_version = 1
	
	def load_prompt( pid: int ) -> None:
		with get_conn( ) as conn:
			cur = conn.execute(
				f"SELECT Name, Text, Version FROM {TABLE} WHERE PromptsId=?",
				(pid,),
			)
			row = cur.fetchone( )
			if row:
				st.session_state.pe_name = row[ 0 ]
				st.session_state.pe_text = row[ 1 ]
				st.session_state.pe_version = row[ 2 ]
	
	# ------------------------------------------------------------------
	# XML / Markdown converters
	# ------------------------------------------------------------------
	def xml_to_md( ):
		st.session_state.pe_text = xml_converter( st.session_state.pe_text )
	
	def md_to_xml( ):
		st.session_state.pe_text = markdown_converter( st.session_state.pe_text )
	
	# ------------------------------------------------------------------
	# Controls (table filters)
	# ------------------------------------------------------------------
	c1, c2, c3, c4 = st.columns( [ 4, 2, 2,  3 ] )
	
	with c1:
		st.text_input( 'Search (Name/Text contains)', key='pe_search' )
	
	with c2:
		st.selectbox( 'Sort by',
			[ 'PromptsId', 'Name', 'Version' ], key='pe_sort_col', )
	
	with c3:
		st.selectbox( 'Direction', [ 'ASC', 'DESC' ], key='pe_sort_dir', )
	
	with c4:
		st.markdown(
			"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
			unsafe_allow_html=True,
		)
		a1, a2, a3 = st.columns( [ 2,
		                           1,
		                           1 ] )
		with a1:
			jump_id = st.number_input(
				"Go to ID",
				min_value=1,
				step=1,
				label_visibility="collapsed",
			)
		with a2:
			if st.button( "Go" ):
				st.session_state.pe_selected_id = int( jump_id )
				load_prompt( int( jump_id ) )
		with a3:
			if st.button( "Undo" ):
				reset_selection( )
	
	# ------------------------------------------------------------------
	# Load prompt table
	# ------------------------------------------------------------------
	where = ""
	params = [ ]
	
	if st.session_state.pe_search:
		where = "WHERE Name LIKE ? OR Text LIKE ?"
		s = f"%{st.session_state.pe_search}%"
		params.extend( [ s,
		                 s ] )
	
	offset = (st.session_state.pe_page - 1) * PAGE_SIZE
	
	query = f"""
        SELECT PromptsId, Name, Text, Version, ID
        FROM {TABLE}
        {where}
        ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
        LIMIT {PAGE_SIZE} OFFSET {offset}
    """
	
	count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"
	
	with get_conn( ) as conn:
		rows = conn.execute( query, params ).fetchall( )
		total_rows = conn.execute( count_query, params ).fetchone( )[ 0 ]
	
	total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
	
	# ------------------------------------------------------------------
	# Prompt table
	# ------------------------------------------------------------------
	table_rows = [ ]
	for r in rows:
		table_rows.append(
			{
					'Selected': r[ 0 ] == st.session_state.pe_selected_id,
					'PromptsId': r[ 0 ],
					'Name': r[ 1 ],
					'Version': r[ 3 ],
					'ID': r[ 4 ],
			}
		)
	
	edited = st.data_editor( table_rows, hide_index=True, use_container_width=True, )
	selected = [ r for r in edited if r.get( "Selected" ) ]
	if len( selected ) == 1:
		pid = selected[ 0 ][ "PromptsId" ]
		if pid != st.session_state.pe_selected_id:
			st.session_state.pe_selected_id = pid
			load_prompt( pid )
	
	# ------------------------------------------------------------------
	# Paging
	# ------------------------------------------------------------------
	p1, p2, p3 = st.columns( [ 0.25, 3.5, 0.25 ] )
	with p1:
		if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
			st.session_state.pe_page -= 1
	with p2:
		st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )
	with p3:
		if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
			st.session_state.pe_page += 1
	
	st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ------------------------------------------------------------------
	# Converter controls
	# ------------------------------------------------------------------
	with st.expander( 'XML ↔ Markdown Converter', expanded=False ):
		b1, b2 = st.columns( 2 )
		with b1:
			st.button( 'Convert XML → Markdown', on_click=xml_to_md )
		with b2:
			st.button( 'Convert Markdown → XML', on_click=md_to_xml )
	
	# ------------------------------------------------------------------
	# Create / Edit Prompt
	# ------------------------------------------------------------------
	with st.expander( 'Create / Edit Prompt', expanded=True ):
		st.text_input(
			'PromptsId',
			value=st.session_state.pe_selected_id or "",
			disabled=True,
		)
		st.text_input( 'Name', key='pe_name' )
		st.text_area( 'Text', key='pe_text', height=260 )
		st.number_input( 'Version', min_value=1, key='pe_version' )
		
		c1, c2, c3 = st.columns( 3 )
		
		with c1:
			if st.button( "Save Changes" if st.session_state.pe_selected_id else "Create Prompt" ):
				with get_conn( ) as conn:
					if st.session_state.pe_selected_id:
						conn.execute(
							f"""
                            UPDATE {TABLE}
                            SET Name=?, Text=?, Version=?
                            WHERE PromptsId=?
                            """,
							(
									st.session_state.pe_name,
									st.session_state.pe_text,
									st.session_state.pe_version,
									st.session_state.pe_selected_id,
							),
						)
					else:
						conn.execute(
							f"""
                            INSERT INTO {TABLE} (Name, Text, Version)
                            VALUES (?, ?, ?)
                            """,
							(
									st.session_state.pe_name,
									st.session_state.pe_text,
									st.session_state.pe_version,
							),
						)
					conn.commit( )
				st.success( 'Saved.' )
				reset_selection( )
		
		with c2:
			if st.session_state.pe_selected_id and st.button( 'Delete' ):
				with get_conn( ) as conn:
					conn.execute(
						f'DELETE FROM {TABLE} WHERE PromptsId=?',
						(st.session_state.pe_selected_id,),
					)
					conn.commit( )
				reset_selection( )
				st.success( 'Deleted.' )
		
		with c3:
			if st.button( 'Clear Selection' ):
				reset_selection( )

# ==============================================================================
# DATA MODE
# ==============================================================================
elif mode == 'Data Export':
	st.subheader( '🚀  Export' )
	st.markdown( '' )
	
	# -----------------------------------
	# Prompt export (System Instructions)
	st.caption( 'System Prompt' )
	
	export_format = st.radio( 'Export Format', options=[ 'XML-Delimited', 'Markdown' ],
		horizontal=True, help='Choose how system instructions should be exported.' )
	prompt_text: str = st.session_state.get( 'system_prompt', '' )
	
	if export_format == 'Markdown':
		try:
			export_text: str = xml_converter( prompt_text )
			export_filename: str = 'Buddy_Instructions.md'
		except Exception as exc:
			st.error( f'Markdown conversion failed: {exc}' )
			export_text = ''
			export_filename = ''
	else:
		export_text = prompt_text
		export_filename = 'Buddy_System_Instructions.xml'
	
	st.download_button(
		label='Download System Instructions',
		data=export_text,
		file_name=export_filename,
		mime='text/plain',
		disabled=not bool( export_text.strip( ) )
	)
	
	# -----------------------------
	# Existing chat history export
	st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	st.markdown( '###### Chat History' )
	
	hist = load_history( )
	md_history = '\n\n'.join(
		[ f'**{role.upper( )}**\n{content}' for role, content in hist ]
	)
	
	st.download_button( 'Download Chat History (Markdown)', md_history,
		'buddy_chat.md', mime='text/markdown' )
	
	buf = io.BytesIO( )
	pdf = canvas.Canvas( buf, pagesize=LETTER )
	y = 750
	
	for role, content in hist:
		pdf.drawString( 40, y, f'{role.upper( )}: {content[ :90 ]}' )
		y -= 14
		if y < 50:
			pdf.showPage( )
			y = 750
	
	pdf.save( )
	
	st.download_button( 'Download Chat History (PDF)', buf.getvalue( ),
		'buddy_chat.pdf', mime='application/pdf' )

# ======================================================================================
# Footer — Fixed Bottom Status Bar (Desktop-style)
# ======================================================================================
st.markdown(
	"""
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

# ---- Fixed footer container
st.markdown(
	"""
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(17, 17, 17, 0.95);
		border-top: 1px solid #2a2a2a;
		padding: 6px 16px;
		font-size: 0.85rem;
		color: #9aa0a6;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

# ---- Resolve active model by mode
_mode_to_model_key = {
		'Text': 'text_model',
		'Images': 'image_model',
		'Audio': 'audio_model',
		'Embeddings': 'embed_model',
}

provider_val = st.session_state.get( "provider", "—" )
mode_val = mode or "—"

active_model = st.session_state.get(
	_mode_to_model_key.get( mode, "" ),
	None,
)

# ---- Build right-side (mode-gated)
right_parts = [ ]

if active_model is not None:
	right_parts.append( f'Model: {active_model}' )

if mode == 'Text':
	temperature = st.session_state.get( 'temperature' )
	top_p = st.session_state.get( 'top_p' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )

elif mode == 'Images':
	size = st.session_state.get( 'image_size' )
	aspect = st.session_state.get( 'image_aspect' )
	
	if aspect is not None:
		right_parts.append( f'Aspect: {aspect}' )
	elif size is not None:
		right_parts.append( f'Size: {size}' )

elif mode == 'Audio':
	task = st.session_state.get( 'audio_task' )
	if task is not None:
		right_parts.append( f'Task: {task}' )

elif mode == 'Embeddings':
	method = st.session_state.get( 'embed_method' )
	if method is not None:
		right_parts.append( f'Method: {method}' )

right_text = " · ".join( right_parts ) if right_parts else "—"

# ---- Render footer
st.markdown(
	f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{provider_val} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """,
	unsafe_allow_html=True,
)
