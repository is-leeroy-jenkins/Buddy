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
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONSTANTS
# ==============================================================================
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )

FAVICON = r'resources/favicon.ico'

CRS = r'https://www.congress.gov/crs-appropriations-status-table'

LOGO = r'resources/buddy_logo.ico'

BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"

HEADING = "#007AFC"

APP_TITLE = 'Buddy'

APP_SUBTITLE = 'AI for Budget Analysts'

PROMPT_ID = 'pmpt_697f53f7ddc881938d81f9b9d18d6136054cd88c36f94549'

PROMPT_VERSION = '8'

VECTOR_STORE_IDS = ['vs_712r5W5833G6aLxIYIbuvVcK', 'vs_697f86ad98888191b967685ae558bfc0']

FILE_IDS = [ 'file-Wd8G8pbLSgVjHur8Qv4mdt', 'file-WPmTsHFYDLGHbyERqJdyqv', 'file-DW5TuqYoEfqFfqFFsMXBvy',
		'file-U8ExiB6aJunAeT6872HtEU', 'file-FHkNiF6Rv29eCkAWEagevT', 'file-XsjQorjtffHTWjth8EVnkL' ]

TEXT_TYPES = { 'output_text' }

MARKDOWN_HEADING_PATTERN: re.Pattern[ str ] = re.compile( r"^##\s+(?P<title>.+?)\s*$" )

XML_BLOCK_PATTERN: re.Pattern[ str ] = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>",
	re.DOTALL )

DB_PATH = "stores/sqlite/Data.db"

DEFAULT_CTX = 4096

CPU_CORES = multiprocessing.cpu_count( )

ANALYST = '❓'

BUDDY = '🧠'

# ==============================================================================
# UTILITIES
# ==============================================================================
def xml_converter( text: str ) -> str:
	"""
		Purpose:
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
		Parameters:
			text (str):
				Prompt text containing XML-like opening and closing tags.
	
		Returns:
			str:
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
			Convert Markdown-formatted system instructions back into
			XML-delimited prompt text by treating level-2 headings (##)
			as section delimiters.
	
		Parameters:
			markdown (str):
				Markdown text using '##' headings to indicate sections.
	
		Returns:
			str:
				XML-delimited text suitable for storage in the prompt database.
				
	"""
	lines: List[ str ] = markdown.splitlines( )
	output: List[ str ] = [ ]
	current_tag: Optional[ str ] = None
	buffer: List[ str ] = [ ]
	
	def flush( ) -> None:
		"""
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

def image_to_base64( path: str ) -> str:
	with open( path, "rb" ) as f:
		return base64.b64encode( f.read( ) ).decode( )

def chunk_text( text: str, size: int = 1200, overlap: int = 200 ) -> List[ str ]:
	chunks, i = [ ], 0
	while i < len( text ):
		chunks.append( text[ i:i + size ] )
		i += size - overlap
	return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def style_headings( text: str, color: str ) -> str:
	return re.sub(
		r"^##\s+(.*)$",
		rf'<h2 style="color:{color};">\1</h2>',
		text,
		flags=re.MULTILINE
	)

def get_dod_css( light_mode: bool ) -> str:
	if not light_mode:
		# =========================
		# DARK MODE
		# =========================
		return """
        <style>
        /* Headings */
        .stChatMessage .stMarkdown h1,
        .stChatMessage .stMarkdown h2,
        .stChatMessage .stMarkdown h3 {
            color: rgb(0, 120, 252) !important;
        }

        /* Body text */
        .stChatMessage .stMarkdown p,
        .stChatMessage .stMarkdown li,
        .stChatMessage .stMarkdown span {
            color: rgb(200, 200, 200) !important;
        }

        /* Strong */
        .stChatMessage .stMarkdown strong {
            color: rgb(220, 220, 220) !important;
        }

        /* Links */
        .stChatMessage .stMarkdown a {
            color: rgb(0, 120, 252) !important;
            text-decoration: underline;
        }

        /* Blockquotes */
        .stChatMessage .stMarkdown blockquote {
            color: rgb(200, 200, 200) !important;
            border-left: 4px solid rgb(0, 120, 252);
        }
        </style>
        """
	else:
		# =========================
		# LIGHT MODE
		# =========================
		return """
        <style>
        /* Headings */
        .stChatMessage .stMarkdown h1,
        .stChatMessage .stMarkdown h2,
        .stChatMessage .stMarkdown h3 {
            color: rgb(0, 120, 252) !important;
        }

        /* Body text */
        .stChatMessage .stMarkdown p,
        .stChatMessage .stMarkdown li,
        .stChatMessage .stMarkdown span {
            color: rgb(45, 45, 45) !important;
        }

        /* Strong */
        .stChatMessage .stMarkdown strong {
            color: rgb(20, 20, 20) !important;
        }

        /* Links */
        .stChatMessage .stMarkdown a {
            color: rgb(0, 84, 166) !important;
            text-decoration: underline;
        }

        /* Blockquotes */
        .stChatMessage .stMarkdown blockquote {
            color: rgb(45, 45, 45) !important;
            border-left: 4px solid rgb(0, 120, 252);
        }
        </style>
        """

# ==============================================================================
# Configuration
# ==============================================================================
client = OpenAI( )

# ==============================================================================
# Page Setup
# ==============================================================================
AVATARS = { 'user': ANALYST, 'assistant': BUDDY, }

st.logo( LOGO, size='large', link=CRS )

st.set_page_config( page_title=APP_TITLE, layout="wide",
	page_icon=FAVICON, initial_sidebar_state='collapsed' )

st.markdown( DOD_CSS, unsafe_allow_html=True )

st.caption( APP_SUBTITLE )

# ==============================================================================
# Session State
# ==============================================================================
def init_state( ) -> None:
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

init_state( )

def reset_state( ) -> None:
	st.session_state.chat_history = [ ]
	st.session_state.last_answer = ""
	st.session_state.last_sources = [ ]
	st.session_state.last_analysis = {
			'tables': [ ],
			'files': [ ],
			'text': [ ],
	}

# ==============================================================================
# Normalization Helpers 
# ==============================================================================
def normalize( obj: Any ) -> Dict[ str, Any ]:
	if isinstance( obj, dict ):
		return obj
	if hasattr( obj, 'model_dump' ):
		return obj.model_dump( )
	return {
			k: getattr( obj, k )
			for k in dir( obj )
			if not k.startswith( "_" ) and not callable( getattr( obj, k ) )
	}

# ==============================================================================
# Response Extractors
# ==============================================================================
def extract_answer( response ) -> str:
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
	sources: List[ Dict[ str, Any ] ] = [ ]
	
	if not response or not getattr( response, 'output', None ):
		return sources
	
	for item in response.output:
		t = getattr( item, 'type', None )
		
		# -------------------------
		# Web search (citations)
		# -------------------------
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

# --------------------------------------------------------------------------------------
# Intent Prefix
# --------------------------------------------------------------------------------------
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

# ==============================================================================
# Database
# ==============================================================================
def ensure_db( ) -> None:
	Path( "stores/sqlite" ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( DB_PATH ) as conn:
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
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute(
			"INSERT INTO chat_history (role, content) VALUES (?, ?)",
			(role, content)
		)

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( DB_PATH ) as conn:
		return conn.execute(
			"SELECT role, content FROM chat_history ORDER BY id"
		).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

# ==============================================================================
# Prompt DB helpers
# ==============================================================================
def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn
		)
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE Name=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute(
			"INSERT INTO Prompts (Name, Text, Version, ID) VALUES (?, ?, ?, ?)",
			(data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ])
		)

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ], pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

# ==============================================================================
# Loaders
# ==============================================================================
@st.cache_resource
def load_embedder( ) -> SentenceTransformer:
	return SentenceTransformer( "all-MiniLM-L6-v2" )

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
	st.header( "" )
	st.divider( )
	dod_mode = st.toggle(
		"DoD Light Mode",
		value=False,
		help="Toggle DoD light-mode text styling (background unchanged)."
	)
	st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	st.subheader( '⚙️ AI Settings' )
	st.radio( 'Execution Mode',
			options=[ 'Standard', 'Guidance Only', 'Analysis Only' ],
			index=[ 'Standard',  'Guidance Only', 'Analysis Only' ].index( st.session_state.execution_mode ),
			key='execution_mode', )
	
	if st.button( 'Clear Chat' ):
		reset_state( )
		st.rerun( )
	
	st.divider( )
	st.subheader( '🧠 Mind Controls' )
	
	# ----------
	# Init
	ensure_db( )
	embedder = load_embedder( )
	st.session_state.setdefault( 'messages', load_history( ) )
	st.session_state.setdefault( 'system_prompt', '' )
	st.session_state.setdefault( 'basic_docs', [ ] )
	st.session_state.setdefault( 'use_semantic', False )
	st.session_state.setdefault( 'selected_prompt_id', None )
	st.session_state.setdefault( 'pending_system_prompt_name', None )

# ======================================================================================
# TAB LIST
# ======================================================================================
tab_chat,  tab_export = st.tabs([ '🧠 Chat',  '💻 Data Export' ] )

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

# =============================================================================
# CHAT TAB
# =============================================================================
with tab_chat:
	st.subheader( "" )
	
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
		with st.chat_message( "assistant", avatar=BUDDY ):
			try:
				with st.spinner( "Running prompt..." ):
					response = client.responses.create(
						prompt={
								"id": PROMPT_ID,
								"version": PROMPT_VERSION,
						},
						input=[
								{
										"role": "user",
										"content": [
												{
														"type": "input_text",
														"text": user_input,
												}
										],
								}
						],
						tools=[
								{
										"type": "file_search",
										"vector_store_ids": VECTOR_STORE_IDS,
								},
								{
										"type": "web_search",
										"filters": {
												"allowed_domains": [
														"congress.gov",
														"google.com",
														"gao.gov",
														"omb.gov",
														"defense.gov",
												]
										},
										"search_context_size": "medium",
										"user_location": {
												"type": "approximate"
										},
								},
								{
										"type": "code_interpreter",
										"container": {
												"type": "auto",
												"file_ids": [
														"file-Wd8G8pbLSgVjHur8Qv4mdt",
														"file-WPmTsHFYDLGHbyERqJdyqv",
														"file-DW5TuqYoEfqFfqFFsMXBvy",
														"file-U8ExiB6aJunAeT6872HtEU",
														"file-FHkNiF6Rv29eCkAWEagevT",
														"file-XsjQorjtffHTWjth8EVnkL",
												],
										},
								},
						],
						include=[
								"web_search_call.action.sources",
								"code_interpreter_call.outputs",
						],
						store=True,
					)
				
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
					st.markdown(
						f"<div class='buddy-response'>{output_text}</div>",
						unsafe_allow_html=True
					)
				else:
					st.warning( "No text response returned by the prompt." )
				
				# -------------------------------
				# Persist minimal chat history
				# -------------------------------
				st.session_state.chat_history.append(
					{
							"role": "user",
							"content": user_input }
				)
				st.session_state.chat_history.append(
					{
							"role": "assistant",
							"content": output_text }
				)
			
			except Exception as e:
				st.error( "An error occurred while running the prompt." )
				st.exception( e )

# ==============================================================================
# Data Export Tab
# ==============================================================================
with tab_export:
	st.subheader( '' )
	st.markdown( '###### Export' )
	
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

