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
import sqlite3
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance
import base64
import fitz
import io
from pathlib import Path
import plotly.express as px
import multiprocessing
import os
import sqlite3
import time
import typing_extensions
import tiktoken
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
	VectorStores
)

from gemini import ( Chat, Images, Files, Embeddings, Transcription, TTS, Translation, VectorStores )

from grok import ( Chat, Images, Files, Transcription, TTS, Translation, VectorStores )

# ==============================================================================
# RESPONSE/CHAT UTILITIES
# ==============================================================================
def extract_response_text( response: object ) -> str:
	"""
		
		Purpose:
		--------
		Safely extract assistant text from a Responses API object.
	
		Parameters:
		-----------
		response (object): The response returned from the OpenAI client.
	
		Returns:
		--------
		str: Concatenated assistant text output. Empty string if none found.
		
	"""
	if response is None:
		return ""
	
	output = getattr( response, "output", None )
	if not output or not isinstance( output, list ):
		return ""
	
	text_chunks: list[ str ] = [ ]
	
	for item in output:
		if not hasattr( item, "type" ):
			continue
		
		if item.type == "message":
			content = getattr( item, "content", None )
			if not content or not isinstance( content, list ):
				continue
			
			for part in content:
				if getattr( part, "type", None ) == "output_text":
					text = getattr( part, "text", "" )
					if text:
						text_chunks.append( text )
	
	return "".join( text_chunks ).strip( )

def convert_xml( text: str ) -> str:
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

def markdown_converter( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )
	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

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

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int = 400 ) -> list[ str ]:
	"""
		
		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed
	
		Parameters
		----------
		text: str
		max_tokens: int
	
		Returns
		-------
		list[str]
		
	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
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
	
	if 'chat_messages' not in st.session_state:
		st.session_state.chat_messages = [ ]
		
	if 'execution_mode' not in st.session_state:
		st.session_state.execution_mode = 'Standard'
		
	for k in ( 'audio_system_instructions',
				'image_system_instructions',
				'docqna_system_instructions',
				'text_system_instructions' ):
		st.session_state.setdefault( k, "" )

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

def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, "model_dump" ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def extract_answer( response: Any ) -> str:
	"""
	
		Purpose:
		_________
		Parses-out answer text from a structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response expected to contain an `output` attribute.
		
		Returns:
		---------
		str
			Concatenated assistant text or empty string.
	
	"""
	texts: List[ str ] = [ ]
	
	if response is None:
		return ''
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return ''
	
	for item in output:
		if item is None:
			continue
		
		item_type = getattr( item, 'type', None )
		
		# ---------------------------------------
		# Direct text items
		# ---------------------------------------
		if item_type in TEXT_TYPES:
			text = getattr( item, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				texts.append( text )
			continue
		
		# ---------------------------------------
		# Nested content blocks
		# ---------------------------------------
		content = getattr( item, 'content', None )
		if not isinstance( content, list ):
			continue
		
		for block in content:
			if block is None:
				continue
			
			block_type = getattr( block, 'type', None )
			if block_type in TEXT_TYPES:
				text = getattr( block, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					texts.append( text )
	
	return '\n'.join( texts ).strip( )

def extract_sources( response: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		_________
		Parses-out sources from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		List[ Dict[ str, Any ] ]
			List of normalized source dictionaries.
	
	"""
	sources: List[ Dict[ str, Any ] ] = [ ]
	
	if response is None:
		return sources
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return sources
	
	for item in output:
		if item is None:
			continue
		
		t = getattr( item, 'type', None )
		
		# ------------------------------------------------
		# Web search
		# ------------------------------------------------
		if t == 'web_search_call':
			action = getattr( item, 'action', None )
			raw = getattr( action, 'sources', None ) if action else None
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for src in raw:
				s = normalize( src )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'title' ), 'snippet': s.get( 'snippet' ),
						'url': s.get( 'url' ), 'files_id': None, } )
		
		# ------------------------------------------------
		# File search (vector store)
		# ------------------------------------------------
		elif t == 'file_search_call':
			raw = getattr( item, 'results', None )
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for r in raw:
				s = normalize( r )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'file_name' ) or s.get( 'title' ),
						'snippet': s.get( 'text' ), 'url': None, 'files_id': s.get( 'files_id' ), } )
	
	return sources

def extract_analysis( response: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		_________
		Parses-out code interpreter artifacts from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		Dict[ str, Any ]
			Dictionary containing tables, files, and text artifacts.
	
	"""
	artifacts: Dict[ str, Any ] = {
			'tables': [ ],
			'files': [ ],
			'text': [ ] }
	
	if response is None:
		return artifacts
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return artifacts
	
	for item in output:
		if item is None:
			continue
		
		if getattr( item, 'type', None ) != 'code_interpreter_call':
			continue
		
		outputs = getattr( item, 'outputs', None )
		if not isinstance( outputs, (list, tuple) ):
			continue
		
		for out in outputs:
			if out is None:
				continue
			
			out_type = getattr( out, 'type', None )
			
			if out_type == 'table':
				normalized = normalize( out )
				artifacts[ 'tables' ].append( normalized )
			
			elif out_type == 'file':
				normalized = normalize( out )
				artifacts[ 'files' ].append( normalized )
			
			elif out_type in TEXT_TYPES:
				text = getattr( out, 'text', None )
				if isinstance( text, str ) and text.strip( ):
					artifacts[ 'text' ].append( text )
	
	return artifacts

def save_temp( upload ) ->  str | None:
	"""
		Purpose:
		--------
		Save a Streamlit UploadedFile object to a temporary file on disk
		and return the filesystem path.
	
		Parameters:
		-----------
		upload : streamlit.runtime.uploaded_file_manager.UploadedFile
			Uploaded file object from st.file_uploader.
	
		Returns:
		--------
		str | None
			Path to the temporary file, or None if invalid input.
	"""
	if upload is None:
		return None
	
	try:
		_, ext = os.path.splitext( upload.name )
		ext = ext or ""
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			tmp.write( upload.getbuffer( ) )
			tmp_path = tmp.name
		
		return tmp_path
	except Exception:
		return None

def _extract_usage_from_response( resp: Any ) -> Dict[ str, int ]:
	"""
	
		Purpose:
		_________
		Extract token usage from a response object/dict.
		Returns dict with prompt_tokens, completion_tokens, total_tokens.
		Defensive: returns zeros if not present.
		
	"""
	usage = { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, }
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

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( "SELECT role, content FROM chat_history ORDER BY id" ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

def format_results( results ):
	formatted_results = ''
	for result in results.data:
		formatted_result = f"<li> '{result.name}'"
		formatted_results += formatted_result + "</li>"
	return f"<p>{formatted_results}</p>"

def count_tokens( text: str ) -> int:
	"""
		
		Purpose
		----------
		Returns the number of tokens in a text string.
		
		Parmeters
		-----------
		string : str
		encoding_name : str
		
		Return
		------------
		int
		
	"""
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	num_tokens = len( encoding.encode( text ) )
	return num_tokens

# ==============================================================================
# PROMPT ENGINEERING UTILITIES
# ==============================================================================
def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Caption" ], data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ], pid)
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

DM_DB_PATH = os.path.join( 'stores', 'sqlite', 'Data.db' )
os.makedirs( os.path.dirname( DM_DB_PATH ), exist_ok=True )

# ==============================================================================
# DATABASE UTILITIES
# ==============================================================================
def initialize_database( ) -> None:
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

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( DM_DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int=0 ) -> pd.DataFrame:
	query = f'SELECT rowid, * FROM "{table}"'
	if limit:
		query += f" LIMIT {limit} OFFSET {offset}"
	with create_connection( ) as conn:
		return pd.read_sql_query( query, conn )

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( "Invalid column name." )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=',  '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line',
			'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def dm_create_table_from_df( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}')
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if "int" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if "float" in dtype_str:
		return "REAL"
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if "bool" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if "datetime" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if "category" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return "TEXT"

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( "Table name required." )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( "Invalid table name." )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ "name" ]
		col_type = col[ "type" ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ "primary_key" ]:
			definition += " PRIMARY KEY"
			if col[ "auto_increment" ] and col_type == "INTEGER":
				definition += " AUTOINCREMENT"
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""	
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ( 'insert ', 'update ', 'delete ', 'drop ', 'alter ',
			'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ' )
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )

	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'

	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute(f'PRAGMA index_list("{table}");').fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};')
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]		
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )		
		row = \
		{ 
				'column': col, 'dtype': str( series.dtype ),
				'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
				'distinct_%': round( (distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
		}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ "min" ] = series.min( )
			row[ "max" ] = series.max( )
			row[ "mean" ] = series.mean( )
		else:
			row[ "min" ] = None
			row[ "max" ] = None
			row[ "mean" ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( "Table and column required." )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

# ======================================================================================
#  PROVIDER UTILITIES
# ======================================================================================
def get_provider_module( ):
	provider = st.session_state.get( 'provider' )
	module_name = cfg.PROVIDERS.get( provider )
	return __import__( module_name )

def get_chat_module( ):
	"""

		Purpose:
		-------
		Returns a Chat() instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Chat( )

def get_tts_module( ):
	"""

		Purpose:
		-------
		Returns a Text to Speech instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.TTS( )

def get_images_module( ):
	"""

		Purpose:
		-------
		Returns an Images instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Images( )

def get_embeddings_module( ):
	"""

		Purpose:
		-------
		Returns an Embeddings instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Embeddings( )

def get_translation_module( ):
	"""

		Purpose:
		-------
		Returns a Translation instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Translation( )

def get_transcription_module( ):
	"""

		Purpose:
		-------
		Returns a Transcription instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Transcription( )

def get_files_module( ):
	"""

		Purpose:
		-------
		Returns a Files instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.Files( )

def get_vectorstores_module( ):
	"""

		Purpose:
		-------
		Returns an Images instance for the currently selected provider.
		Ensures Gemini / Grok functionality is not bypassed.
		
	"""
	provider_module = get_provider_module( )
	return provider_module.VectorStores( )

def _provider( ):
	return st.session_state.get( 'provider' )

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
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', chat.model_options )
	return chat.model_options

# ---------------- IMAGES ----------------
def image_model_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'image_model_options', image.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'image_model_options', image.model_options )
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', image.model_options )
	return image.model_options

def image_size_or_aspect_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'aspect_options', image.size_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'aspect_options', image.size_options )
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', image.size_options )
	return image.size_options

# ---------------- AUDIO ----------------
def audio_model_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'audio_model_options', transcriber.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'audio_model_options', transcriber.model_options )
	return transcriber.model_options

def audio_language_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'language_options', transcriber.language_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'language_options', transcriber.language_options )
	return transcriber.language_options

# ---------------- EMBEDDINGS ----------------
def embedding_model_options( embed ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'embedding_model_options', embed.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'embedding_model_options', embed.model_options )
	return embed.model_options

# -------------DOC Q&A ----------------------
def route_document_query( prompt: str ) -> str:
	source = st.session_state.get( 'doc_source' )
	active_docs = st.session_state.get( 'docqna_active_docs', [ ] )
	doc_bytes = st.session_state.get( 'doc_bytes', { } )
	
	if not source:
		return 'No document source selected.'
	
	if not active_docs:
		return 'No document selected.'
	
	# --------------------------------------------------
	# LOCAL DOCUMENT → Chat (single or multi)
	# --------------------------------------------------
	if source == 'uploadlocal':
		chat = get_chat_module( )
		
		# Single document
		if len( active_docs ) == 1:
			name = active_docs[ 0 ]
			file_bytes = doc_bytes.get( name )
			
			if not file_bytes:
				return 'Document content not available.'
			
			text = extract_text_from_bytes( file_bytes )
			
			full_prompt = f"""
				{instructions}
				
				Use the following document to answer the question.
				Be precise and cite relevant portions when possible.
				
				DOCUMENT:
				{text}
				
				QUESTION:
				{prompt}
				"""
			return chat.generate_text( prompt=full_prompt )
		
		# Multi-document injection
		combined_text = ""
		
		for name in active_docs:
			file_bytes = doc_bytes.get( name )
			if not file_bytes:
				continue
			
			text = extract_text_from_bytes( file_bytes )
			
			combined_text += f"\n\n===== DOCUMENT: {name} =====\n\n{text}\n"
		
		if not combined_text.strip( ):
			return 'No readable document content available.'
		
		full_prompt = f"""
			{instructions}
			
			You are analyzing multiple documents.
			
			Use the content below to answer the question.
			If multiple documents are relevant, compare them.
			Cite document names when possible.
			
			DOCUMENT SET:
			{combined_text}
			
			QUESTION:
			{prompt}
			"""
		
		return chat.generate_text( prompt=full_prompt )
	
	# --------------------------------------------------
	# FILES API → Files class
	# --------------------------------------------------
	if source == "filesapi":
		files = get_files_module( )
		
		# Single file search
		if len( active_docs ) == 1:
			return files.search( prompt, active_docs[ 0 ] )
		
		# Multi-file survey
		return files.survey( prompt )
	
	# --------------------------------------------------
	# VECTOR STORE → VectorStores class
	# --------------------------------------------------
	if source == 'vectorstore':
		vectorstores = get_vectorstores_module( )
		
		# Single store
		if len( active_docs ) == 1:
			return vectorstores.search( prompt, active_docs[ 0 ] )
		
		# Multi-store aggregation
		responses = [ ]
		for store_id in active_docs:
			result = vectorstores.search( prompt, store_id )
			if result:
				responses.append( result )
		
		if not responses:
			return 'No results found across selected vector stores.'
		
		return "\n\n".join( responses )
	
	return 'Unsupported document source.'

def extract_text_from_bytes( file_bytes: bytes ) -> str:
	"""
		Extracts text from PDF or text-based documents.
	"""
	try:
		import fitz  # PyMuPDF		
		doc = fitz.open( stream=file_bytes, filetype="pdf" )
		text = ""
		for page in doc:
			text += page.get_text( )
		return text.strip( )
	
	except Exception:
		try:
			return file_bytes.decode( errors="ignore" )
		except Exception:
			return ""

def summarize_active_document( ) -> str:
	"""
		Uses the routing layer to summarize the currently active document.
	"""
	doc_instructions = st.session_state.get( "doc_instructions", "" )
	summary_prompt = """
		Provide a clear, structured summary of this document.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points (if any)
		- Policy implications (if applicable)
		
		Be precise and concise.
		"""
	if doc_instructions:
		summary_prompt = f"{doc_instructions}\n\n{summary_prompt}"
	
	return route_document_query( summary_prompt.strip( ) )

# ==============================================================================
# Page Setup / Configuration
# ==============================================================================
openai_client = OpenAI(  )
st.session_state[ 'openai_client' ] = openai_client
AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BUDDY, }
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON, 
	initial_sidebar_state='collapsed' )

st.caption( cfg.APP_SUBTITLE )
inject_response_css( )
init_state( )

# ======================================================================================
# SESSION STATE PARAMETER DEFINITIONS
# ======================================================================================
if 'api_keys' not in st.session_state:
	st.session_state.api_keys = { 'GPT': None, 'Grok': None, 'Gemini': None, }

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
	default = cfg.GEMINI_API_KEY
	if default:
		st.session_state.gemini_api_key = default
		os.environ[ 'GEMINI_API_KEY' ] = default

if st.session_state.groq_api_key == '':
	default = cfg.GROQ_API_KEY
	if default:
		st.session_state.groq_api_key = default
		os.environ[ 'GROQ_API_KEY' ] = default

if st.session_state.google_api_key == '':
	default = cfg.GOOGLE_API_KEY
	if default:
		st.session_state.google_api_key = default
		os.environ[ 'GOOGLE_API_KEY' ] = default

if st.session_state.xai_api_key == '':
	default = cfg.XAI_API_KEY
	if default:
		st.session_state.xai_api_key = default
		os.environ[ 'XAI_API_KEY' ] = default

if 'provider' not in st.session_state or st.session_state[ 'provider' ] is None:
	st.session_state[ 'provider' ] = 'GPT'

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Chat'

if 'files' not in st.session_state:
	st.session_state.files: List[ str ] = [ ]
	
#----------MODEL PARAMETERS --------------------------------
if 'model' not in st.session_state:
	st.session_state.model = ''

if 'text_model' not in st.session_state:
	st.session_state[ 'text_model' ] = ''
	
if 'image_model' not in st.session_state:
	st.session_state[ 'image_model' ] = ''
	
if 'audio_model' not in st.session_state:
	st.session_state[ 'audio_model' ] = ''
	
if 'embedding_model' not in st.session_state:
	st.session_state[ 'embedding_model' ] = ''

if 'docqna_model' not in st.session_state:
	st.session_state[ 'docqna_model' ] = ''

if 'files_model' not in st.session_state:
	st.session_state[ 'files_model' ] = ''

if 'stores_model' not in st.session_state:
	st.session_state[ 'stores_model' ] = ''

if 'tts_model' not in st.session_state:
	st.session_state[ 'tts_model' ] = ''

if 'transcription_model' not in st.session_state:
	st.session_state[ 'transcription_model' ] = ''

if 'translation_model' not in st.session_state:
	st.session_state[ 'translation_model' ] = ''

# --------SYSTEM PARAMETERS----------------------
if 'instructions' not in st.session_state:
	st.session_state[ 'instructions' ] = ''
	
if 'text_system_instructions' not in st.session_state:
	st.session_state[ 'text_system_instructions' ] = ''

if 'image_system_instructions' not in st.session_state:
	st.session_state[ 'image_system_instructions' ] = ''

if 'audio_system_instructions' not in st.session_state:
	st.session_state[ 'audio_system_instructions' ] = ''

if 'docqna_system_instructions' not in st.session_state:
	st.session_state[ 'docqna_systems_instructions' ] = ''

#--------CHAT-GENERATION PARAMETERS--------------------

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0
	
if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0
	
if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'execution_mode' not in st.session_state:
	st.session_state[ 'execution_mode' ] = ''

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'last_sources' not in st.session_state:
	st.session_state[ 'last_sources' ] = [ ]

# --------TEXT-GENERATION PARAMETERS--------------------
if 'text_max_tokens' not in st.session_state:
	st.session_state[ 'text_max_tokens' ] = 0

if 'text_max_calls' not in st.session_state:
	st.session_state[ 'text_max_calls' ] = 0

if 'text_temperature' not in st.session_state:
	st.session_state[ 'text_temperature' ] = 0.0

if 'text_top_percent' not in st.session_state:
	st.session_state[ 'text_top_percent' ] = 0.0

if 'text_frequency_penalty' not in st.session_state:
	st.session_state[ 'text_frequency_penalty' ] = 0.0

if 'text_presense_penalty' not in st.session_state:
	st.session_state[ 'text_presense_penalty' ] = 0.0

if 'text_parallel_tools' not in st.session_state:
	st.session_state[ 'text_parallel_tools' ] = False

if 'text_background' not in st.session_state:
	st.session_state[ 'text_background' ] = False

if 'text_store' not in st.session_state:
	st.session_state[ 'text_store' ] = False

if 'text_stream' not in st.session_state:
	st.session_state[ 'text_stream' ] = False

if 'text_response_format' not in st.session_state:
	st.session_state[ 'text_response_format' ] = ''

if 'text_tool_choice' not in st.session_state:
	st.session_state[ 'text_tool_choice' ] = ''

if 'text_reasoning' not in st.session_state:
	st.session_state[ 'text_reasoning' ] = ''

if 'text_content' not in st.session_state:
	st.session_state[ 'text_content' ] = ''

if 'text_stops' not in st.session_state:
	st.session_state[ 'text_stops' ] = [ ]

if 'text_include' not in st.session_state:
	st.session_state[ 'text_include' ] = [ ]

if 'text_domains' not in st.session_state:
	st.session_state[ 'text_domains' ] = [ ]

if 'text_input' not in st.session_state:
	st.session_state[ 'text_input' ] = [ ]

if 'text_tools' not in st.session_state:
	st.session_state.text_tools: List[ Dict[ str, Any ] ] = [ ]

if 'text_messages' not in st.session_state:
	st.session_state[ 'text_messages' ] = [ ]

# --------IMAGE-GENERATION PARAMETERS--------------------
if 'image_max_tokens' not in st.session_state:
	st.session_state[ 'image_max_tokens' ] = 0
	
if 'image_max_calls' not in st.session_state:
	st.session_state[ 'image_max_calls' ] = 0

if 'image_temperature' not in st.session_state:
	st.session_state[ 'image_temperature' ] = 0.0

if 'image_top_percent' not in st.session_state:
	st.session_state[ 'image_top_percent' ] = 0.0

if 'image_frequency_penalty' not in st.session_state:
	st.session_state[ 'image_frequency_penalty' ] = 0.0

if 'image_presense_penalty' not in st.session_state:
	st.session_state[ 'image_presense_penalty' ] = 0.0

if 'image_parallel_tools' not in st.session_state:
	st.session_state[ 'image_parallel_tools' ] = False

if 'image_background' not in st.session_state:
	st.session_state[ 'image_background' ] = False

if 'image_store' not in st.session_state:
	st.session_state[ 'image_store' ] = False

if 'image_stream' not in st.session_state:
	st.session_state[ 'image_stream' ] = False

if 'image_tool_choice' not in st.session_state:
	st.session_state[ 'image_tool_choice' ] = ''

if 'image_reasoning' not in st.session_state:
	st.session_state[ 'image_reasoning' ] = ''

if 'image_quality' not in st.session_state:
	st.session_state[ 'image_quality' ] = ''

if 'image_format' not in st.session_state:
	st.session_state[ 'image_format' ] = ''

if 'image_response_format' not in st.session_state:
	st.session_state[ 'image_response_format' ] = ''

if 'image_input' not in st.session_state:
	st.session_state[ 'image_input' ] = ''

if 'image_include' not in st.session_state:
	st.session_state[ 'image_include' ] = [ ]

if 'image_tools' not in st.session_state:
	st.session_state.image_tools: List[ Dict[ str, Any ] ] = [ ]

if 'image_messages' not in st.session_state:
	st.session_state.image_messages: List[ Dict[ str, Any ] ] = [ ]

if 'image_domains' not in st.session_state:
	st.session_state[ 'image_domains' ] = [ ]

if 'image_content' not in st.session_state:
	st.session_state[ 'image_content' ] = [ ]

# ------- IMAGE-SPECIFIC PARAMETER---------------
if 'image_mode' not in st.session_state:
	st.session_state[ 'image_mode' ] = ''
	
if 'image_size' not in st.session_state:
	st.session_state[ 'image_size' ] = ''

if 'image_style' not in st.session_state:
	st.session_state[ 'image_style' ] = ''

if 'image_detail' not in st.session_state:
	st.session_state[ 'image_detail' ] = ''

if 'image_backcolor' not in st.session_state:
	st.session_state[ 'image_backcolor' ] = ''

if 'image_quality' not in st.session_state:
	st.session_state[ 'image_quality' ] = ''

if 'image_output' not in st.session_state:
	st.session_state[ 'image_output' ] = ''

if 'image_url' not in st.session_state:
	st.session_state[ 'image_url' ] = ''

# --------AUDIO-GENERATION PARAMETERS--------------------
if 'audio_max_tokens' not in st.session_state:
	st.session_state[ 'audio_max_tokens' ] = 0
	
if 'audio_temperature' not in st.session_state:
	st.session_state[ 'audio_temperature' ] = 0.0

if 'audio_top_percent' not in st.session_state:
	st.session_state[ 'audio_top_percent' ] = 0.0

if 'audio_frequency_penalty' not in st.session_state:
	st.session_state[ 'audio_frequency_penalty' ] = 0.0

if 'audio_presense_penalty' not in st.session_state:
	st.session_state[ 'audio_presense_penalty' ] = 0.0

if 'audio_background' not in st.session_state:
	st.session_state[ 'audio_background' ] = False

if 'audio_store' not in st.session_state:
	st.session_state[ 'audio_store' ] = False

if 'audio_stream' not in st.session_state:
	st.session_state[ 'audio_stream' ] = False

if 'audio_tool_choice' not in st.session_state:
	st.session_state[ 'audio_tool_choice' ] = ''

if 'audio_reasoning' not in st.session_state:
	st.session_state[ 'audio_reasoning' ] = ''

if 'audio_response_format' not in st.session_state:
	st.session_state[ 'audio_response_format' ] = { }

if 'audio_input' not in st.session_state:
	st.session_state[ 'audio_input' ] = [ ]

if 'audio_input' not in st.session_state:
	st.session_state[ 'audio_input' ] = [ ]

if 'audio_stops' not in st.session_state:
	st.session_state[ 'audio_stops' ] = [ ]

if 'audio_includes' not in st.session_state:
	st.session_state[ 'audio_includes' ] = [ ]

if 'audio_tools' not in st.session_state:
	st.session_state.audio_tools: List[ Dict[ str, Any ] ] = [ ]

if 'audio_messages' not in st.session_state:
	st.session_state.audio_messages: List[ Dict[ str, Any ] ] = [ ]

#-------AUDIO-SECIFIC PARAMETERS--------------
if 'audio_file' not in st.session_state:
	st.session_state[ 'audio_file' ] = ''

if 'sample_rate' not in st.session_state:
	st.session_state[ 'sample_rate' ] = 0

if 'language' not in st.session_state:
	st.session_state[ 'language' ] = ''

if 'voice' not in st.session_state:
	st.session_state[ 'voice' ] = ''

if 'start_time' not in st.session_state:
	st.session_state[ 'start_time' ] = 0.0

if 'end_time' not in st.session_state:
	st.session_state[ 'end_time' ] = 0.0

if 'loop' not in st.session_state:
	st.session_state[ 'loop' ] = False
	
if 'play' not in st.session_state:
	st.session_state[ 'play' ] = False

if 'audio_format' not in st.session_state:
	st.session_state[ 'audio_format' ] = ''

# ------- EMBEDDING-SPECIFIC PARAMETERS ----------------------
if 'embeddings_input_text' not in st.session_state:
	st.session_state[ 'embeddings_input_text' ] = ''

if 'embeddings_dimensions' not in st.session_state:
	st.session_state[ 'embeddings_dimensions' ] = 0

if 'embeddings_encoding_format' not in st.session_state:
	st.session_state[ 'embeddings_encoding_format' ] = ''

if 'embeddings_batch_size' not in st.session_state:
	st.session_state[ 'embeddings_batch_size' ] = 0

if 'embeddings_method' not in st.session_state:
	st.session_state[ 'embeddings_method' ] = ''

# ------- FILES-SPECIFIC PARAMETERS --------------------------
if 'files_purpose' not in st.session_state:
	st.session_state[ 'files_purpose' ] = ''

if 'files_type' not in st.session_state:
	st.session_state[ 'files_type' ] = ''

if 'files_id' not in st.session_state:
	st.session_state[ 'files_id' ] = ''

if 'files_url' not in st.session_state:
	st.session_state[ 'files_url' ] = ''
	
# -------- VECTORSTORES-GENERATION PARAMETERS --------------------

if 'stores_temperature' not in st.session_state:
	st.session_state[ 'stores_temperature' ] = 0.0

if 'stores_top_percent' not in st.session_state:
	st.session_state[ 'stores_top_percent' ] = 0.0

if 'stores_max_tokens' not in st.session_state:
	st.session_state[ 'stores_max_tokens' ] = 0

if 'stores_frequency_penalty' not in st.session_state:
	st.session_state[ 'stores_frequency_penalty' ] = 0.0

if 'stores_presense_penalty' not in st.session_state:
	st.session_state[ 'stores_presense_penalty' ] = 0.0

if 'stores_max_calls' not in st.session_state:
	st.session_state[ 'stores_max_calls' ] = 0

if 'stores_tool_choice' not in st.session_state:
	st.session_state[ 'stores_tool_choice' ] = ''

if 'stores_response_format' not in st.session_state:
	st.session_state[ 'stores_response_format' ] = ''

if 'stores_reasoning' not in st.session_state:
	st.session_state[ 'stores_reasoning' ] = ''

if 'stores_parallel_tools' not in st.session_state:
	st.session_state[ 'stores_parallel_tools' ] = False

if 'stores_background' not in st.session_state:
	st.session_state[ 'stores_background' ] = False

if 'stores_store' not in st.session_state:
	st.session_state[ 'stores_store' ] = False

if 'stores_stream' not in st.session_state:
	st.session_state[ 'stores_stream' ] = False

if 'stores_input' not in st.session_state:
	st.session_state[ 'stores_input' ] = [ ]

if 'stores_tools' not in st.session_state:
	st.session_state[ 'stores_tools' ] = [ ]

if 'stores_messages' not in st.session_state:
	st.session_state[ 'stores_messages' ] = [ ]

if 'stores_stops' not in st.session_state:
	st.session_state[ 'stores_stops' ] = [ ]

if 'stores_include' not in st.session_state:
	st.session_state[ 'stores_include' ] = [ ]

# ------- VECTORSTORES-SPECIFIC PARAMETERS -------------------

if 'stores_id' not in st.session_state:
	st.session_state[ 'stores_id' ] = ''

#------- DOCQA-SPECIFIC PARAMATERS  ---------------------------
if 'docqna_files' not in st.session_state:
	st.session_state[ 'docqna_files' ] = [ ]
	
if 'docqna_uploaded' not in st.session_state:
	st.session_state[ 'docqna_uploaded' ] = ''

if 'docqna_messages' not in st.session_state:
	st.session_state.docqna_messages = [ ]

if 'docqna_active_docs' not in st.session_state:
	st.session_state.docqna_active_docs = [ ]

if 'docqna_source' not in st.session_state:
	st.session_state.docqna_source = ''

if 'docqna_multi_mode' not in st.session_state:
	st.session_state.docqna_multi_mode = False

# ------- TOKEN PARAMATERS  ---------------------------
if 'last_answer' not in st.session_state:
	st.session_state.last_answer = ''

if 'last_sources' not in st.session_state:
	st.session_state.last_sources = [ ]

if 'last_analysis' not in st.session_state:
	st.session_state.last_analysis = {
			'tables': [ ],
			'docqna_files': [ ],
			'text': [ ],
	}

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0, }

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
	                                 'total_tokens': 0, }

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
	provider = st.session_state.get( 'provider'  )

	style_subheaders( )
	st.subheader( '' )
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	provider = st.selectbox( 'Select API', list( cfg.PROVIDERS.keys( ) ),
		index=list( cfg.PROVIDERS.keys( ) ).index( st.session_state.get( 'provider', 'GPT' ) ) )
	
	st.session_state[ 'provider' ] = provider
	logo_path = cfg.LOGO_MAP.get( provider )
	if logo_path and os.path.exists( logo_path ):
		logo_path = cfg.LOGO_MAP.get( provider )
		st.logo( logo_path, size='large', link=cfg.CRS )
	
	#-----API KEY Expander------------------------------
	with st.expander( label='Keys:', icon='🔑', expanded=False ):
		openai_key = st.text_input( 'OpenAI API Key', type='password',
			value=st.session_state.openai_api_key or '',
			help='Overrides OPENAI_API_KEY from config.py for this session only.' )
		
		gemini_key = st.text_input( 'Gemini API Key', type='password',
			value=st.session_state.gemini_api_key or '',
			help='Overrides GEMINI_API_KEY from config.py for this session only.' )
		
		groq_key = st.text_input( 'Groq API Key', type='password',
			value=st.session_state.groq_api_key or '',
			help='Overrides GROQ_API_KEY from config.py for this session only.' )
		
		google_key = st.text_input('Google API Key', type='password',
			value=st.session_state.google_api_key or '',
			help='Overrides GOOGLE_API_KEY from config.py for this session only.' )
		
		xai_key = st.text_input( 'xAi API Key', type='password',
			value=st.session_state.xai_api_key or '',
			help='Overrides XAI_API_KEY from config.py for this session only.' )
		
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
			
	if st.button( 'Clear Chat' ):
		reset_state( )
		st.rerun( )
		
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	if provider == 'Gemini':
		mode = st.sidebar.radio( 'Select Mode', cfg.GEMINI_MODES, index=0 )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	elif provider == 'Grok':
		mode = st.sidebar.radio( 'Select Mode', cfg.GROK_MODES, index=0 )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	else:
		mode = st.sidebar.radio( 'Select Mode', cfg.GPT_MODES, index=0 )

# =============================================================================
# CHAT MODE
# =============================================================================
if mode == 'Chat':
	st.subheader( "💬 Chat Completions", help=cfg.CHAT_COMPLETIONS )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	chat_number = st.session_state.get( 'number', 0 )
	chat_top_p = st.session_state.get( 'top_percent', 0.0 )
	chat_freq = st.session_state.get( 'frequency_penalty', 0.0 )
	chat_presense = st.session_state.get( 'presense_penalty', 0.0 )
	chat_temperature = st.session_state.get( 'temperature', 0.0 )
	chat_background = st.session_state.get( 'background', False )
	chat_stream = st.session_state.get( 'stream', False )
	chat_store = st.session_state.get( 'store', False )
	chat_model = st.session_state.get( 'model', '' )
	chat_format = st.session_state.get( 'response_format', '' )
	chat_input = st.session_state.get( 'input', [ ] )
	chat_reasoning = st.session_state.get( 'reasoning', '' )
	chat_choice = st.session_state.get( 'tool_choice', '' )
	chat_messages = st.session_state.get( 'messages', [ ] )
	execution_mode = st.session_state.get( 'execution_mode', '' )
	chat_history = st.session_state.get( 'chat_history', [ ] )
	
	# ------------------------------------------------------------------
	# Sidebar — Text Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.text( '⚙️  Chat Settings' )
		st.radio( 'Execution Mode', options=[ 'Standard', 'Guidance Only', 'Analysis Only' ],
			index=[ 'Standard', 'Guidance Only', 'Analysis Only' ].index( st.session_state.execution_mode ),
			key='execution_mode', )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.25,  3.5,  0.25 ] )
	with center:
		user_input = st.chat_input( 'Have a Planning, Programming, or Budget Execution question?' )
		if user_input:
			# -------------------------------
			# Render user message
			# -------------------------------
			with st.chat_message( 'user', avatar=cfg.ANALYST ):
				st.markdown( user_input )
			
			# -------------------------------
			# Run prompt
			# -------------------------------
			with st.chat_message( 'assistant', avatar=cfg.BUDDY ):
				try:
					chat = OpenAI( api_key=cfg.OPENAI_API_KEY )
					with st.spinner( 'Running prompt...' ):
						response = chat.responses.create(
							prompt={ 'id': cfg.PROMPT_ID, 'version': cfg.PROMPT_VERSION, },
							input=[ { 'role': 'user', 'content': [ { 'type': 'input_text',
							                                         'text': user_input, } ], } ],
							tools=[ { 'type': 'file_search', 'vector_store_ids': cfg.GPT_VECTORSTORES, },
									{ 'type': 'web_search', 'filters': { 'allowed_domains': cfg.GPT_DOMAINS, },
											'search_context_size': 'medium',
											'user_location': { 'type': 'approximate' },
									},
									{ 'type': 'code_interpreter',
									  'container': { 'type': 'auto', 'file_ids': cfg.GPT_FILES, },
									}, ],
							include=[ 'web_search_call.action.sources',
							          'code_interpreter_call.outputs', ], store=True, )
					sources = st.session_state.get( "last_sources", [ ] )
					if sources:
						st.markdown( "#### Sources" )
						for i, src in enumerate( sources, 1 ):
							url = src.get( "url" )
							title = src.get( "title" ) or src.get( "file_name" ) or f"Source {i}"
							
							if url:
								st.markdown( f"- [{title}]({url})" )
							elif src.get( "file_id" ):
								st.markdown( f"- {title} _(Vector Store File: `{src[ 'files_id' ]}`)_" )
					
					# -------------------------------
					# Extract and render text output
					# -------------------------------
					output_text = ""
					for item in response.output:
						if item.type == 'message':
							for part in item.content:
								if part.type == 'output_text':
									output_text += part.text
					
					if output_text.strip( ):
						st.markdown( output_text )
					else:
						st.warning( 'No text response returned by the prompt.' )
					
					# -------------------------------
					# Persist minimal chat history
					# -------------------------------
					st.session_state.chat_history.append({'role': 'user', 'content': user_input})
					st.session_state.chat_history.append({'role':'assistant', 'content':output_text})
				except Exception as e:
					st.error( 'An error occurred while running the prompt.' )
					st.exception( e )

# ======================================================================================
# TEXT MODE
# ======================================================================================
elif mode == 'Text':
	st.subheader( "💬 Text Generation", help=cfg.TEXT_GENERATION )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	text_number = st.session_state.get( 'text_number', 0 )
	text_max_calls = st.session_state.get( 'text_max_calls', 0 )
	text_tokens = st.session_state.get( 'text_max_tokens', 0 )
	text_top_percent = st.session_state.get( 'text_top_percent', 0.0 )
	text_freq = st.session_state.get( 'text_frequency_penalty', 0.0 )
	text_presense = st.session_state.get( 'text_presense_penalty', 0.0 )
	text_temperature = st.session_state.get( 'text_temperature', 0.0 )
	text_stream = st.session_state.get( 'text_stream', False )
	text_parallel_tools = st.session_state.get( 'text_parallel_tools', False )
	text_store = st.session_state.get( 'text_store', False )
	text_background = st.session_state.get( 'text_background', False )
	text_model = st.session_state.get( 'text_model', '' )
	text_reasoning = st.session_state.get( 'text_reasoning', '' )
	text_response_format = st.session_state.get( 'text_response_format', '' )
	text_choice = st.session_state.get( 'text_tool_choice', '' )
	text_content = st.session_state.get( 'text_content', '' )
	text_tools = st.session_state.get( 'text_tools', [ ] )
	text_include = st.session_state.get( 'text_include', [ ] )
	text_domains = st.session_state.get( 'text_domains', [ ] )
	text_stops = st.session_state.get( 'text_stops', [ ] )
	text_input = st.session_state.get( 'text_input', [ ] )
	text_messages = st.session_state.get( 'text_messages', [ ] )
	text = provider_module.Chat( )
	
	for key in [ 'text_domains', 'text_stops', 'text_tools',
	             'text_includes', 'text_input',  ]:
		if key in st.session_state and isinstance( st.session_state[ key ], list ):
			del st.session_state[ key ]
		
	# ------------------------------------------------------------------
	# Sidebar — Text Settings
	# ------------------------------------------------------------------
	with (st.sidebar):
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		if st.session_state.get( 'do_clear_instructions' ):
			st.session_state[ 'text_system_instructions' ] = ''
			st.session_state[ 'instructions_last_loaded' ] = ''
			st.session_state[ 'do_clear_instructions' ] = False
		
		# ------------------------------------------------------------------
		# Expander — Text LLM Configuration
		# ------------------------------------------------------------------
		with st.expander( label='LLM Configuration', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Text Generation Model Parameter
			# ------------------------------------------------------------------
			with st.expander( label='Model Settings', expanded=False, width='stretch' ):
					llm_c1, llm_c2, llm_c3, llm_c4  = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
						border=True, gap='medium' )
					
					with llm_c1:
						model_options = list( text.model_options )
						set_text_model = st.selectbox( label='Select Model', options=model_options,
							help='REQUIRED. Text Generation model used by the AI',
							key='text_model', placeholder='Options' )
						
						text_model = st.session_state[ 'text_model' ]
					
					with llm_c2:
						include_options = list( text.include_options )
						set_text_includes = st.multiselect( label='Include:', options=include_options,
							key='text_include', help=cfg.INCLUDE, placeholder='Options' )
						
						text_includes = st.session_state[ 'text_include' ]
					
					with llm_c3:
						set_text_domains = st.text_input( label='Allowed Domains', key='text_domains_input',
						value=','.join( st.session_state.get( 'text_domains', [ ] )  ),
						help=cfg.ALLOWED_DOMAINS, width='stretch', placeholder='Enter Domains' )
						
						text_domains = [ d.strip( ) for d in set_text_domains.split( ',' )
								if d.strip( ) ]
						
						st.session_state[ 'text_domains' ] = text_domains

					with llm_c4:
						reasoning_options = list( text.reasoning_options )
						set_text_reasoning = st.selectbox( label='Reasoning Effort:',
							options=reasoning_options, key='text_reasoning',
							help=cfg.REASONING, placeholder='Options'  )
						
						text_reasoning = st.session_state[ 'text_reasoning' ]
					
					if st.button( label='Reset', key='text_model_reset', width='stretch' ):
						# ----------------------------------------------------------
						# Remove Model Settings session keys
						# ----------------------------------------------------------
						for key in [ 'text_model', 'text_include', 'text_domains', 'text_stops',
						             'text_reasoning', 'text_domains_input' ]:
							if key in st.session_state:
								del st.session_state[ key ]
						
						st.rerun( )
	
			# ------------------------------------------------------------------
			# Text Generation Inference Parameters
			# ------------------------------------------------------------------
			with st.expander( label='Inference Settings', expanded=False, width='stretch' ):
				prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns( [ 0.20, 0.20, 0.20, 0.20, 0.20 ],
					border=True, gap='xxsmall' )
				
				with prm_c1:
					set_text_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						step=0.01, help=cfg.TOP_P, key='text_top_percent' )
					text_top_percent = st.session_state[ 'text_top_percent' ]
				
				with prm_c2:
					set_text_freq = st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'text_frequency_penalty', 0.0 )),
						step=0.01, help=cfg.FREQUENCY_PENALTY, key='text_frequency_penalty' )
					text_fequency = st.session_state[ 'text_frequency_penalty' ]
				
				with prm_c3:
					set_text_presense = st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'text_presense_penalty', 0.0 ) ),
						step=0.01, help=cfg.PRESENCE_PENALTY, key='text_presense_penalty' )
					text_presense = st.session_state[ 'text_presense_penalty' ]
				
				with prm_c4:
					set_text_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
						value=float( st.session_state.get( 'text_temperature', 0.0 ) ), step=0.01,
						help=cfg.TEMPERATURE, key='text_temperature' )
					
					text_temperature = st.session_state[ 'text_temperature' ]
				
				with prm_c5:
					set_text_number = st.number_input( label='Number', min_value=1, max_value=4,
						step=1, help='Optional. Upper limit on the responses returned by the model',
						key='text_number' )
					text_number = st.session_state[ 'text_number' ]
				
				if st.button( label='Reset', key='text_inference_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Inference Settings session keys
					# ----------------------------------------------------------
					for key in [ 'text_top_percent', 'text_frequency_penalty',
					             'text_presense_penalty', 'text_temperature', 'text_number', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )

			# ------------------------------------------------------------------
			# Text Generation Tool Options
			# ------------------------------------------------------------------
			with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='medium' )
				
				with tool_c1:
					set_text_parallel = st.toggle( label='Allow Parallel', key='text_parallel_tools',
						help=cfg.PARALLEL_TOOL_CALLS )
					text_parallel_tools = st.session_state[ 'text_parallel_tools' ]
				
				with tool_c2:
					set_text_calls = st.number_input( label='Max Calls', min_value=0, max_value=5,
						step=1, help=cfg.MAX_TOOL_CALLS, key='text_max_tools' )
					text_max_tools = st.session_state[ 'text_max_tools' ]
				
				with tool_c3:
					choice_options = text.choice_options
					set_text_choice = st.multiselect( label='Tool Choice:', options=choice_options,
						key='text_tool_choice', help=cfg.CHOICE, placeholder='Options' )
					text_tool_choice = st.session_state[ 'text_tool_choice' ]
				
				with tool_c4:
					tool_options = text.tool_options
					set_text_tools = st.multiselect( label='Tools:', options=tool_options,
						key='text_tools', help=cfg.TOOLS, placeholder='Options' )
					text_tools = st.session_state[ 'text_tools' ]
				
				if st.button( label='Reset', key='text_tools_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Tool Settings session keys
					# ----------------------------------------------------------
					for key in [ 'text_parallel_tools', 'text_max_tools',
					             'text_tool_choice', 'text_tools', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )

			# ------------------------------------------------------------------
			# Expander — Text Generation Response
			# ------------------------------------------------------------------
			with st.expander( label='Response Settings', expanded=False, width='stretch' ):
					resp_c1, resp_c2, resp_c3, resp_c4, resp_c5 = st.columns(
						[0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
					
					with resp_c1:
						set_text_stream = st.toggle( label='Stream', key='text_stream',
							help=cfg.STREAM )
						
						text_stream = st.session_state[ 'text_stream' ]
						
					with resp_c2:
						set_text_store = st.toggle( label='Store', key='text_store',
							help=cfg.STORE )
						
						text_store = st.session_state[ 'text_store' ]
						
					with resp_c3:
						set_text_background = st.toggle( label='Background', key='text_background',
							 help=cfg.BACKGROUND_MODE )
						
						text_background = st.session_state[ 'text_background' ]
						
					with resp_c4:
						set_text_stops = st.text_input( label='Stop Sequences', key='text_stops',
							help=cfg.STOP_SEQUENCE, width='stretch', placeholder='Enter Stops' )
						
						text_stops = [ d.strip( ) for d in set_text_domains.split( ',' )
						               if d.strip( ) ]
					
					with resp_c5:
						set_text_tokens = st.number_input( label='Max Tokens', min_value=0, max_value=100000,
							step=1000, help=cfg.MAX_OUTPUT_TOKENS, key='text_max_tokens' )
						
						text_tokens = st.session_state[ 'text_max_tokens' ]
					
					if st.button( label='Reset', key='text_response_reset', width='stretch' ):
						# ----------------------------------------------------------
						# Remove Response Settings session keys
						# ----------------------------------------------------------
						for key in [ 'text_stream', 'text_store', 'text_background',
								'text_stops', 'text_max_tokens', ]:
							if key in st.session_state:
								del st.session_state[ key ]
						# If using separated UI key for stops
						if 'text_stops_input' in st.session_state:
							del st.session_state[ 'text_stops_input' ]
						
						st.rerun( )

		# ------------------------------------------------------------------
		# Expander — Text System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]

			with in_left:
				st.text_area( label='Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='text_system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'text_system_instructions' ] = text
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, placeholder='',
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'text_system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( label='Clear Instructions', width='stretch', on_click=_on_clear )
			
		
		# ----------- MESSAGES ---------------------------------
		if st.session_state[ 'text_messages' ] is not None:
			for msg in st.session_state.text_messages:
				with st.chat_message( msg[ 'role' ], avatar="" ):
					st.markdown( msg[ 'content' ] )
		
		if provider_name == 'GPT':
			prompt = st.chat_input( 'Ask ChatGPT…' )
		elif provider_name == 'Grok':
			prompt = st.chat_input( 'Ask Grok…' )
		elif provider_name == 'Gemini':
			prompt = st.chat_input( 'Ask Gemini…' )
		else:
			prompt = None
		
		if prompt is not None:
			st.session_state.text_messages.append( { 'role': 'user', 'content': prompt } )
			with st.chat_message( 'assistant', avatar="" ):
				gen_kwargs = { }
				
				with st.spinner( 'Thinking…' ):
					gen_kwargs[ 'model' ] = st.session_state[ 'text_model' ]
					gen_kwargs[ 'top_percent' ] = st.session_state[ 'text_top_percent' ]
					gen_kwargs[ 'background' ] = st.session_state[ 'text_background' ]
					gen_kwargs[ 'max_tokens' ] = st.session_state[ 'text_max_tokens' ]
					gen_kwargs[ 'frequency' ] = st.session_state[ 'text_frequency_penalty' ]
					gen_kwargs[ 'presence' ] = st.session_state[ 'text_presense_penalty' ]
					
					if st.session_state[ 'text_stops' ]:
						gen_kwargs[ 'stops' ] = st.session_state[ 'text_stops' ]
					
					response = None
					
					try:
						mdl = str( gen_kwargs[ 'text_model' ] )
						if mdl.startswith( 'gpt-5' ):
							response = chat.generate_text( prompt=prompt, model=gen_kwargs[ 'text_model' ] )
						else:
							response = chat.generate_text( )
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
						response = None
					
					if response is not None and str( response ).strip( ):
						st.markdown( response )
						st.session_state.text_messages.append( { 'role': 'assistant', 'content': response } )
					else:
						st.error( 'Generation Failed!.' )
			
						try:
							_update_token_counters( getattr( text, 'response', None ) or response )
						except Exception:
							pass
			
# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == "Images":
	st.subheader( '📷 Images API', help=cfg.IMAGES_API )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	image_number = st.session_state.get( 'image_number', 0 )
	image_max_calls = st.session_state.get( 'image_max_calls', 0 )
	image_max_tokens = st.session_state.get( 'image_max_tokens', 0 )
	image_top_percent = st.session_state.get( 'image_top_percent', 0.0 )
	image_frequency = st.session_state.get( 'image_frequency_penalty', 0.0 )
	image_presense = st.session_state.get( 'image_presense_penalty', 0.0 )
	image_temperature = st.session_state.get( 'image_temperature', 0.0 )
	image_stream = st.session_state.get( 'image_stream', False )
	image_store = st.session_state.get( 'image_store', False )
	image_parallel_calls = st.session_state.get( 'image_parallel_calls', False )
	image_background = st.session_state.get( 'image_background', False )
	image_model = st.session_state.get( 'image_model', '' )
	image_response_format = st.session_state.get( 'image_response_format', '' )
	image_detail = st.session_state.get( 'image_detail', '' )
	image_style = st.session_state.get( 'image_style', '' )
	image_quality = st.session_state.get( 'image_quality', '' )
	image_backcolor = st.session_state.get( 'image_backcolor', '' )
	image_content = st.session_state.get( 'image_content', '' )
	image_size = st.session_state.get( 'image_size', '' )
	image_output = st.session_state.get( 'image_output', '' )
	image_input = st.session_state.get( 'image_input', '' )
	image_mode = st.session_state.get( 'image_mode', '' )
	image_stops = st.session_state.get( 'image_stops', [ ] )
	image_domains = st.session_state.get( 'image_domains', [ ] )
	image_include = st.session_state.get( 'image_include', [ ] )
	image_tools = st.session_state.get( 'image_tools', [ ] )
	image_messages = st.session_state.get( 'image_messages', [ ] )
	image_input = st.session_state.get( 'image_input', [ ] )
	generator = None
	analyzer = None
	editor = None
	# ---------------- Task ----------------
	available_tasks = [ ]
	model_options = [ ]
	image = provider_module.Images( )
	
	for key in [ 'image_domains', 'image_tools', 'image_stops', ]:
		if key in st.session_state and isinstance( st.session_state[ key ], list ):
			del st.session_state[ key ]
		
	# ------------------------------------------------------------------
	# EXPANDER — IMAGE SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'do_clear_instructions' ):
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'clear_image_instructions' ] = False
		st.session_state[ 'do_clear_instructions' ] = False
		
	# ------------------------------------------------------------------
	# Image Main  UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with (center):
		# ------------------------------------------------------------------
		# Expander — Image LLM Configuration
		# ------------------------------------------------------------------
		with st.expander( label='LLM Configuration', icon='🧠', expanded=False, width='stretch' ):
			with st.expander( label='Model Settings', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4, llm_c5 = st.columns( [ 0.20, 0.20, 0.20, 0.20, 0.20 ],
					border=True, gap='xxsmall' )
				
				with llm_c1:
					_modes = [ 'Generation', 'Analysis', 'Editing' ]
					set_image_mode = st.selectbox( label='Image Mode:', options=_modes,
						key='image_mode', help='Available Image API modes',
						placeholder='Options' )
					
					image_mode = st.session_state[ 'image_mode' ]
					
				with llm_c2:
					if st.session_state[ 'image_mode' ] == 'Generation':
						generation = cfg.GPT_GENERATION
						set_image_model = st.selectbox( label='Select Model', options=generation,
						help='REQUIRED. Images Generation model used by the AI', key='image_model',
						placeholder='Options' )
						
						image_model = st.session_state[ 'image_model' ]
						
					elif st.session_state[ 'image_mode' ] == 'Analysis':
						analysis = cfg.GPT_ANALYSIS
						set_image_model = st.selectbox( label='Select Model', options=analysis,
							help='REQUIRED. Images Generation model used by the AI', key='image_model',
							placeholder='Options' )
						
						image_model = st.session_state[ 'image_model' ]
						
					elif st.session_state[ 'image_mode' ] == 'Editing':
						editing = cfg.GPT_EDITING
						set_image_model = st.selectbox( label='Select Model', options=editing,
						help='REQUIRED. Images Generation model used by the AI', key = 'image_model',
						placeholder='Options' )
						image_model = st.session_state[ 'image_model' ]
					else:
						all = cfg.GPT_GENERATION
						set_image_model = st.selectbox( label='Select Model', options=all,
							help='REQUIRED. Images Generation model used by the AI', key='image_model',
							placeholder='Options' )
						
						image_model = st.session_state[ 'image_model' ]
				
				with llm_c3:
					includes = image.include_options
					set_image_include = st.multiselect( label='Include:',
						options=includes, key='image_include',
						help=cfg.INCLUDE, placeholder='Options' )
					
					image_include = st.session_state[ 'image_include' ]
				
				with llm_c4:
					set_image_domains = st.text_input( label='Allowed Domains', key='image_domains',
						placeholder='Enter Domains',
						help=cfg.ALLOWED_DOMAINS, width='stretch' )
					
					image_domains = [ d.strip( ) for d in set_image_domains.split( ',' )
							if d.strip( ) ]
					
					image_domains = st.session_state[ 'image_domains' ]
				
				with llm_c5:
					reasonings = image.reasoning_options
					set_image_reasoning = st.selectbox( label='Reasoning:', placeholder='Options',
						options=reasonings, key='image_reasoning', help=cfg.REASONING )
					
					image_reasoning = st.session_state[ 'image_reasoning' ]
				
				if st.button( label='Reset', key='image_model_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Image Model Settings session keys
					# ----------------------------------------------------------
					for key in [ 'image_model', 'image_include', 'image_domains', 'image_stops',
					              'image_mode', 'image_reasoning' ]:
						if key in st.session_state:
							del st.session_state[ key ]
							
						if 'image_domains_input' in st.session_state:
							del st.session_state[ 'image_domains_input' ]
						
						st.rerun( )
			
			with st.expander( label='Inference Settings', expanded=False, width='stretch' ):
				prm_c1, prm_c2, prm_c3, prm_c4, prm_c5 = st.columns( [ 0.20, 0.20, 0.20, 0.20, 0.20 ],
					border=True, gap='xxsmall' )
				
				with prm_c1:
					set_image_top_p = st.slider( label='Top-P',  key='image_top_percent',
						min_value=0.0, max_value=1.0, step=0.01, help=cfg.TOP_P )
					
					image_top_percent = st.session_state[ 'image_top_percent' ]
				
				with prm_c2:
					set_image_freq = st.slider( label='Frequency Penalty',
						key='image_frequency_penalty',
						min_value=-2.0, max_value=2.0, step=0.01, help=cfg.FREQUENCY_PENALTY )
					
					image_fequency = st.session_state[ 'image_frequency_penalty' ]
				
				with prm_c3:
					set_image_presense = st.slider( label='Presence Penalty',
						key='image_presense_penalty',
						min_value=-2.0, max_value=2.0, step=0.01,  help=cfg.PRESENCE_PENALTY )
					
					image_presense = st.session_state[ 'image_presense_penalty' ]
				
				with prm_c4:
					set_image_temperature = st.slider( label='Temperature',
						key='image_temperature',
						min_value=0.0, max_value=1.0, step=0.01, help=cfg.TEMPERATURE )
					
					image_temperature = st.session_state[ 'image_temperature' ]
				
				with prm_c5:
					set_image_number = st.number_input( label='Number', min_value=0, max_value=100,
						step=1, help='Optional. Upper limit on the responses returned by the model',
						key='image_number' )
					
					image_number = st.session_state[ 'image_number' ]
					
				if st.button( label='Reset', key='image_inference_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Image Inferrence Settings session keys
					# ----------------------------------------------------------
					for key in [ 'image_top_percent', 'image_frequency_penalty',
					             'image_presense_penalty', 'image_temperature',  'image_number' ]:
						if key in st.session_state:
							del st.session_state[ key ]
							
					st.rerun( )

			with st.expander( label='Tool Settings', expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				with tool_c1:
					set_image_parallel = st.toggle( label='Allow Parallel', key='image_parallel_tools',
						help=cfg.PARALLEL_TOOL_CALLS )
					
					image_parallel_tools = st.session_state[ 'image_parallel_tools' ]
				
				with tool_c2:
					set_image_calls = st.number_input( label='Max Tools', min_value=0, max_value=6,
						step=1, help=cfg.MAX_TOOL_CALLS, key='image_max_tools' )
					
					image_max_tools = st.session_state[ 'image_max_tools' ]
				
				with tool_c3:
					choices = image.choice_options
					set_image_choice = st.selectbox( label='Tool Choice:', options=choices,
						key='image_choice', help=cfg.CHOICE, placeholder='Options')
					
					image_tool_choice = st.session_state[ 'image_choice' ]
				
				with tool_c4:
					set_image_tools = st.multiselect( label='Tools:', options=image.tool_options,
						key='image_tools', help=cfg.TOOLS, placeholder='Options' )
					
					image_tools = st.session_state[ 'image_tools' ]
				
				if st.button( label='Reset', key='image_tools_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Image Tool Settings session keys
					# ----------------------------------------------------------
					for key in [ 'image_parallel_tools', 'image_max_tools', 'image_tool_choice',
							'image_tools', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )

			with st.expander( label='Response Settings', expanded=False, width='stretch' ):
				res_one, res_two, res_three, res_four, res_five = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				with res_one:
					set_image_stream = st.toggle( label='Stream', key='image_stream', help=cfg.STREAM )
					
					image_stream = st.session_state[ 'image_stream' ]
				
				with res_two:
					set_image_store = st.toggle( label='Store', key='image_store', help=cfg.STORE )
					
					text_store = st.session_state[ 'image_store' ]
				
				with res_three:
					set_image_background = st.toggle( label='Background', key='image_background',
						help=cfg.BACKGROUND_MODE )
					
					image_background = st.session_state[ 'image_background' ]
				
				with res_four:
					formats = image.format_options
					set_image_reponse = st.selectbox( label='Response Format:',
						options=formats, key='image_response_format',
						help=cfg.IMAGE_RESPONSE, placeholder='Options' )
					
					image_respose_format = st.session_state[ 'image_response_format' ]
				
				with res_five:
					set_image_tokens = st.number_input( label='Max Tokens', min_value=0, max_value=100000,
						step=1000, help=cfg.MAX_OUTPUT_TOKENS, key='image_max_tokens' )
					
					image_tokens = st.session_state[ 'image_max_tokens' ]
				
				if st.button( label='Reset', key='image_response_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Image Response session keys
					# ----------------------------------------------------------
					for key in [ 'image_stream', 'image_store', 'image_background',
							'image_response_format', 'image_max_tokens', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					# If canonical separation used
					if 'image_stops_input' in st.session_state:
						del st.session_state[ 'image_stops_input' ]
					
					st.rerun( )

			with st.expander( label='Visual Settings', expanded=False, width='stretch' ):
				img_c1, img_c2, img_c3, img_c4, img_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				# ------------ Image Detail
				with img_c1:
					details = image.detail_options
					set_image_detail = st.selectbox( label='Image Detail', options=details,
						help='Optional. Image detail', key='image_detail', placeholder='Options' )
					
					image_detail = st.session_state[ 'image_detail' ]
				
				# ------------ Image Style
				with img_c2:
					sizes = image.size_options
					set_image_size = st.selectbox( label='Image Size', options=sizes,
						help='Optional. Image size', key='image_size', placeholder='Options' )
					
					image_size = st.session_state[ 'image_size' ]
				
				# ------------ Image Quality
				with img_c3:
					qualities = image.quality_options
					set_image_quality = st.selectbox( label='Image Quality',
						options=qualities, help='Optional. Image Quality',
						key='image_quality', placeholder='Options' )
					
					image_quality = st.session_state[ 'image_quality' ]
				
				# ------------ Image Backcolor
				with img_c4:
					colors = image.backcolor_options
					set_image_backcolor = st.selectbox( label='Image Backcolor',
						options=colors,
						help=cfg.IMAGE_BACKGROUND, key='image_backcolor', placeholder='Options' )
					
					image_backcolor = st.session_state[ 'image_backcolor' ]
				
				# ------------ Image Output Format
				with img_c5:
					outputs = image.output_options
					set_image_output = st.selectbox( label='Image Format', options=outputs,
						help=cfg.IMAGE_RESPONSE, key='image_output', placeholder='Options' )
					
					image_output = st.session_state[ 'image_output' ]
				
				if st.button( label='Reset', key='image_settings_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Image Response session keys
					# ----------------------------------------------------------
					for key in [ 'image_detail', 'image_backcolor', 'image_style', 'image_quality'
					             'image_size', 'image_output', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Image System Instructions
		# ------------------------------------------------------------------
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( 'Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='image_system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'image_system_instructions' ] = text
			
			with in_right:
				st.selectbox( 'Select Template', prompt_names,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'image_system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( 'Clear Instructions', width='stretch', on_click=_on_clear )
		
		# Image Modes
		tab_gen, tab_analyze, tab_edit = st.tabs( [ 'Generate', 'Analyze', 'Edit' ] )
		with tab_gen:
			prompt = st.chat_input( 'Prompt' )
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
			uploaded_img = st.file_uploader(
				'Upload an image for analysis',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ],
				accept_multiple_files=False,
				key='images_analyze_uploader', )
			
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview', use_column_width=True, )
				
				# Discover available analysis methods on Image object
				available_methods = [ ]
				for candidate in ( 'analyze', 'describe_image', 'describe', 'classify',
							'detect_objects', 'caption', 'image_analysis', ):
					if hasattr( image, candidate ):
						available_methods.append( candidate )
				
				if available_methods:
					chosen_method = st.selectbox( 'Method', available_methods, index=0, )
				else:
					chosen_method = None
					st.info( 'No dedicated image analysis method found on Image object; '
						'attempting generic handlers.' )
				
				chosen_model = st.selectbox( 'Model (analysis)', [ image_model, None ], index=0, )
				
				chosen_model_arg = ( image_model if chosen_model is None else chosen_model )
				if st.button( 'Analyze Image' ):
					with st.spinner( 'Analyzing image…' ):
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
								for fallback in ( 'analyze', 'describe_image', 'describe', 'caption' ):
									if hasattr( image, fallback ):
										func = getattr( image, fallback )
										try:
											analysis_result = func( tmp_path )
											break
										except Exception:
											continue
							
							if analysis_result is None:
								st.warning(
									'No analysis output returned by the available methods.'
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
		
		with tab_edit:
			uploaded_img = st.file_uploader( 'Upload Image for Edit',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ],
				accept_multiple_files=False,
				key='images_edit_uploader',
			)
			
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				
				st.image( uploaded_img, caption='Uploaded image preview', use_column_width=True, )
				
				available_methods = [ ]
				for candidate in ( 'edit', 'describe_image', 'describe', 'classify',
							'detect_objects', 'caption', 'image_edit', ):
					if hasattr( image, candidate ):
						available_methods.append( candidate )
				
				if available_methods:
					chosen_method = st.selectbox( 'Method', available_methods, index=0, )
				else:
					chosen_method = None
					st.info( 'No dedicated image editing method found on Image object;'
					         'attempting generic handlers.')
				
				chosen_model = st.selectbox( 'Model (edit)', [ image_model,  None ], index=0, )
				
				chosen_model_arg = ( image_model if chosen_model is None else chosen_model )
				
				if st.button( 'Edit Image' ):
					with st.spinner( 'Editing image…' ):
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
								for fallback in ( 'analyze', 'describe_image',
											'describe', 'caption', ):
									if hasattr( image, fallback ):
										func = getattr( image, fallback )
										try:
											analysis_result = func( tmp_path )
											break
										except Exception:
											continue
							
							if analysis_result is None:
								st.warning( 'No editing output returned by the available methods.' )
							else:
								if isinstance( analysis_result, (dict, list) ):
									st.json( analysis_result )
								else:
									st.markdown( '**Analysis result:**' )
									st.write( analysis_result )
								
								try:
									_update_token_counters(
										getattr( image, 'response', None )
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
	st.subheader( '🔉 Audio API', help=cfg.AUDIO_API )
	st.divider( )
	# ------------------------------------------------------------------
	# Provider-aware Audio instantiation
	# ------------------------------------------------------------------
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	audio_model = st.session_state.get( 'audio_model', '' )
	audio_format = st.session_state.get( 'audio_response_format', '' )
	audio_top_p = st.session_state.get( 'audio_top_percent', 0.0 )
	audio_freq = st.session_state.get( 'audio_frequency_penalty', 0.0 )
	audio_presense = st.session_state.get( 'audio_presense_penalty', 0.0 )
	audio_number = st.session_state.get( 'audio_number', 0 )
	audio_temperature = st.session_state.get( 'audio_temperature', 0.0 )
	audio_stream = st.session_state.get( 'audio_stream', False )
	audio_store = st.session_state.get( 'audio_store', False )
	audio_input = st.session_state.get( 'audio_input', [ ] )
	audio_reasoning = st.session_state.get( 'audio_reasoning', '' )
	audio_choice = st.session_state.get( 'audio_tool_choice', '' )
	audio_messages = st.session_state.get( 'audio_messages', [ ] )
	audio_background = st.session_state.get( 'audio_background', True )
	audio_file = st.session_state.get( 'audio_file', '')
	audio_rate = st.session_state.get( 'audio_rate', 0 )
	audio_start = st.session_state.get( 'audio_start', 0.0 )
	audio_end = st.session_state.get( 'audio_end', 0.0 )
	audio_loop = st.session_state.get( 'audio_loop', False )
	auto_play = st.session_state.get( 'auto_play', False )
	voice = st.session_state.get( 'voice', None )
	transcriber = None
	translator = None
	tts = None
	if hasattr( provider_module, 'Transcription' ):
		transcriber = provider_module.Transcription( )
	if hasattr( provider_module, 'Translation' ):
		translator = provider_module.Translation( )
	if hasattr( provider_module, 'TTS' ):
		tts = provider_module.TTS( )
	
	# ---------------- Task ----------------
	available_tasks = [ ]
	model_options = [ ]
	language = None
	voice = None
	audio_model = None
	
	if transcriber is not None:
		available_tasks.append( 'Transcribe' )
	if translator is not None:
		available_tasks.append( 'Translate' )
	if tts is not None:
		available_tasks.append( 'Text-to-Speech' )
		
	# ------------------------------------------------------------------
	#  AUDIO SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'do_clear_instructions' ):
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'clear_audio_instructions' ] = False
		st.session_state[ 'do_clear_instructions' ] = False
		
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9,  0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — Audio LLM Configuration
		# ------------------------------------------------------------------
		with st.expander( '🧠 LLM Configuration', expanded=False, width='stretch' ):
			# Expander — Audio Model
			with st.expander( 'Model Options', expanded=False, width='stretch' ):
				aud_one, aud_two, aud_three, aud_four, aud_five = st.columns(
					[ 0.2, 0.2, 0.2, 0.2, 0.2 ], gap='xxsmall', border=True )
				
				# ---------------- Task ---------------
				with aud_one:
					if not available_tasks:
						st.info( 'Audio is not supported by the selected provider.' )
						task = None
					else:
						task = st.selectbox( 'Mode', available_tasks )
				
				# ---------------- Model ---------------
				with aud_two:
					if task == 'Transcribe' and transcriber and hasattr( transcriber, 'model_options' ):
						model_options = transcriber.model_options
					elif task == 'Translate' and translator and hasattr( translator, 'model_options' ):
						model_options = translator.model_options
					elif task == 'Text-to-Speech' and tts and hasattr( tts, 'model_options' ):
						model_options = tts.model_options
					
					if model_options:
						audio_model = st.selectbox( 'Model', model_options,
							index=(model_options.index( st.session_state.get( 'audio_model' ) )
							       if st.session_state.get( 'audio_model' ) in model_options
							       else 0), )
						st.session_state[ 'audio_model' ] = audio_model
				
				# ---------------- Language ----------------
				with aud_three:
					if task in ('Transcribe', 'Translate'):
						obj = transcriber if task == 'Transcribe' else translator
						if obj and hasattr( obj, 'language_options' ):
							language = st.selectbox( 'Language', obj.language_options, )
					
					if task == 'Text-to-Speech' and tts:
						if hasattr( tts, 'voice_options' ):
							voice = st.selectbox( 'Voice', tts.voice_options, )
				
				# ---------------- Sample Rate ----------------
				with aud_four:
					audio_rate = st.selectbox( label='Speed', options=cfg.SAMPLE_RATES )
				
				# ---------------- Format ----------------
				with aud_five:
					if task == 'Transcribe' and transcriber and hasattr( transcriber, 'format_options' ):
						format_options = transcriber.format_options
					elif task == 'Translate' and translator and hasattr( translator, 'format_options' ):
						format_options = translator.format_options
					elif task == 'Text-to-Speech' and tts and hasattr( tts, 'format_options' ):
						format_options = tts.format_options
					
					if format_options:
						audio_format = st.selectbox( 'Format', format_options,
							index=(format_options.index( st.session_state.get( 'audio_format' ) )
							       if st.session_state.get( 'audio_format' ) in format_options
							       else 0), )
						st.session_state[ 'audio_format' ] = audio_format
				
				if st.button( 'Reset', key='audio_model_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Audio Model Settings session keys
					# ----------------------------------------------------------
					for key in [ 'audio_model', 'audio_language', 'audio_voice', 'audio_rate',
							'audio_format', ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )

			# Expander — Audio Inference
			with st.expander( 'Inference Options', expanded=False, width='stretch' ):
				prm_one, prm_two, prm_three, prm_four = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				with prm_one:
					set_audio_top = st.slider( 'Top-P', 0.0, 1.0,
						float( st.session_state.get( 'audio_top_percent', 1.0 ) ), 0.01, help=cfg.TOP_P )
					audio_top_p = st.session_state[ 'audio_top_percent' ]
			
				with prm_two:
					set_audio_frequency = st.slider( 'Frequency Penalty', -2.0, 2.0,
						float( st.session_state.get( 'audio_frequency_penalty', 0.0 ) ),
						0.01, help=cfg.FREQUENCY_PENALTY )
					audio_frequency = st.session_state[ 'audio_frequency_penalty' ]
				
				with prm_three:
					set_audio_presense = st.slider( 'Presence Penalty', -2.0, 2.0,
						float( st.session_state.get( 'audio_presense_penalty', 0.0 ) ),
						0.01, help=cfg.PRESENCE_PENALTY )
					audio_presense = st.session_state[ 'audio_presense_penalty' ]
				
				with prm_four:
					set_audio_temperature = st.slider( 'Temperature', 0.0, 1.0,
						float( st.session_state.get( 'audio_temperature', 0.7 ) ), 0.01,
						help=cfg.TEMPERATURE )
					audio_temperature = st.session_state[ 'audio_temperature' ]
				
				if st.button( 'Reset', key='audio_inference_reset', width='stretch' ):
					# ----------------------------------------------------------
					# Remove Audio Inference session keys
					# ----------------------------------------------------------
					for key in [ 'audio_top_percent', 'audio_temperature' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			# Expander — Audio Response
			with st.expander( 'Response Options', expanded=False, width='stretch' ):
				resp_one, resp_two, resp_three, resp_four, resp_five = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], gap='xxsmall', border=True, )
				
				with resp_one:
					loop = st.toggle( label='Loop Audio', value=False )
					st.session_state[ 'audio_loop' ] = loop
				
				with resp_two:
					play = st.toggle( label='Auto Play', value=False )
					st.session_state[ 'audio_loop' ] = play
				
				with resp_three:
					start = st.number_input( label='Start Time:', min_value=0.0, value=0.0 )
					st.session_state[ 'audio_start' ] = start
				
				with resp_four:
					end = st.number_input( label='End Time:', min_value=0.0, value=0.0 )
					st.session_state[ 'number' ] = end
				
				with resp_five:
					max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=100000,
						value=6048, help=cfg.MAX_OUTPUT_TOKENS )
					st.session_state[ 'max_tokens' ] = int( max_tokens )
				
				st.button( 'Reset', key='audio_response_reset', width='stretch' )
		
		# ------------------------------------------------------------------
		# Expander — Audio System Instructions
		# ------------------------------------------------------------------
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( 'Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='audio_system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'audio_system_instructions' ] = text
			
			with in_right:
				st.selectbox( 'Select Template', prompt_names,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'audio_system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( 'Clear Instructions', width='stretch', on_click=_on_clear )
		
		left_audio, center_audio, right_audio = st.columns( [ 0.33, 0.33, 0.33 ], border=True )
		
		# -----------UPLOAD AUDIO----------------------
		with left_audio:
			if task in ('Transcribe', 'Translate'):
				uploaded = st.file_uploader( 'Upload File', type=[ 'wav', 'mp3', 'm4a', 'flac' ], )
				if uploaded:
					tmp_path = save_temp( uploaded )
					if task == 'Transcribe' and transcriber:
						with st.spinner( 'Transcribing…' ):
							try:
								text = transcriber.transcribe( tmp_path, model=audio_model,
									language=language, )
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
								text = translator.translate( tmp_path, model=audio_model,
									language=language, )
								st.text_area( 'Translation', value=text, height=300 )
								
								try:
									_update_token_counters( getattr( translator, 'response', None ) )
								except Exception:
									pass
							
							except Exception as exc:
								st.error( f'Translation failed: {exc}' )
			
			elif task == 'Text-to-Speech' and tts:
				text = st.text_area( 'Text to synthesize' )
				
				if text and st.button( 'Generate Audio' ):
					with st.spinner( 'Synthesizing speech…' ):
						try:
							audio_bytes = tts.speak( text, model=audio_model, voice=voice, )
							st.audio( audio_bytes )
							
							try:
								_update_token_counters( getattr( tts, "response", None ) )
							except Exception:
								pass
						
						except Exception as exc:
							st.error( f"Text-to-speech failed: {exc}" )
							
		#-----------RECORD AUDIO----------------------
		with center_audio:
			recording = st.audio_input( label='Record Audio', sample_rate=audio_rate)
			
		# -----------PLAY AUDIO----------------------
		with right_audio:
			if audio_file:
				audio_recording = st.audio( data, sample_rate=audio_rate,
					start_time=audio_start, end_time=audio_end, format=audio_format, width='stretch',
					loop=audio_loop, autoplay=auto_play )
			else:
				array = None
	
# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	st.subheader( '⚡ Vector Embeddings', help=cfg.EMBEDDINGS_API )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	embedding_model = st.session_state.get( 'embedding_model' )
	dimensions = st.session_state.get( 'embeddings_dimensions' )
	encoding = st.session_state.get( 'embeddings_encoding_format' )
	input_text = st.session_state.get( 'embeddings_input_text' )
	embedding = provider_module.Embeddings( )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	emb_left, emb_center, emb_right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with emb_center:
		# ------------------------------------------------------------------
		# Expander — Embedding LLM Configuration
		# ------------------------------------------------------------------
		with st.expander( 'Configuration', expanded=False, width='stretch' ):
			llm_c1, llm_c2, llm_c3, llm_c4  = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
				border=True, gap='medium' )
			
			with llm_c1:
				set_embedding_model = st.selectbox( 'Embedding Model:', embedding.model_options,
					help='REQUIRED. Embedding model used by the AI',
					key='embedding_model',
					index=(embedding.model_options.index( st.session_state[ 'embedding_model' ] )
					       if st.session_state.get( 'embedding_model' ) in embedding.model_options else 0), )
				embedding_model = st.session_state[ 'embedding_model' ]
			
			with llm_c2:
				set_encoding_format = st.selectbox( 'Encoding Format:',
					options=embedding.encoding_options, key='embeddings_encoding_format',
					help='REQUIRED: The format to return the embeddings in. float or base64')
				embedding_encoding_format = st.session_state[ 'embeddings_encoding_format' ]
			
			with llm_c3:
				set_embedding_dimensions = st.number_input( 'Dimensions', min_value=0,
					max_value=2048, step=1, key='embeddings_dimensions',
					help='Optional (large models only): An integer between 1 and 2048',
					width='stretch' )
				embedding_dimensions = st.session_state[ 'embeddings_dimensions' ]
			
			with llm_c4:
				st.number_input( 'Chunk Size (tokens)', min_value=50, max_value=2000,
					step=50, key='embedding_chunk_size',
					help='Maximum tokens per chunk for embedding segmentation.' )

			if st.button( 'Reset', key='embedding_reset', width='stretch' ):
				
				# ----------------------------------------------------------
				# Remove Embedding Configuration session keys
				# ----------------------------------------------------------
				for key in [ 'embedding_model', 'embeddings_dimensions', 'embeddings_encoding_format',
						'embeddings_input_text', ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )

		# ------------------------------------------------------------------
		# Main UI — Embedding execution (unchanged behavior)
		# ------------------------------------------------------------------
		set_input_text = st.text_area( 'Text to embed', key='embeddings_input_text' )
		btn_left, btn_right = st.columns( [ 0.50, 0.50 ] )
		
		with btn_left:
			embed_clicked = st.button( 'Embed', width='stretch', key='embedding_set' )
			if embed_clicked and input_text and input_text.strip( ):
				with st.spinner( 'Embedding…' ):
					try:
						# ----------------------------------------------------------
						# Normalize + Chunk
						# ----------------------------------------------------------
						chunk_size = st.session_state.get( 'embedding_chunk_size', 400 )
						normalized_text = normalize_text( input_text )
						chunks = chunk_text( normalized_text, max_tokens=chunk_size )
						
						# ----------------------------------------------------------
						# Create Embeddings
						# ----------------------------------------------------------
						if embedding_dimensions is not None:
							vectors = embedding.create(
								text=chunks,
								model=embedding_model,
								dimensions=embedding_dimensions
							)
						else:
							vectors = embedding.create(
								text=chunks,
								model=embedding_model
							)
						
						# ----------------------------------------------------------
						# Persist Results
						# ----------------------------------------------------------
						st.session_state[ 'embeddings' ] = vectors
						st.session_state[ 'embedding_chunks' ] = chunks
						
						# ----------------------------------------------------------
						# Display Summary
						# ----------------------------------------------------------
						try:
							if isinstance( vectors, list ) and vectors and isinstance( vectors[ 0 ], list ):
								vector_dimension = len( vectors[ 0 ] )
								st.write( 'Chunks:', len( vectors ) )
								st.write( 'Vector dimension:', vector_dimension )
							elif isinstance( vectors, list ):
								st.write( 'Vector dimension:', len( vectors ) )
							else:
								st.write( 'Vector result type:', type( vectors ) )
						except Exception:
							st.write( 'Vector length:', len( vectors ) )
						
						# ----------------------------------------------------------
						# Token Counters
						# ----------------------------------------------------------
						try:
							_update_token_counters( getattr( embedding, 'response', None ) )
						except Exception:
							pass
					
					except Exception as exc:
						st.error( f'Embedding failed: {exc}' )
		
		with btn_right:
			if st.button( 'Reset', width='stretch', key='input_text_reset' ):
				# ----------------------------------------------------------
				# Clear Embedding State
				# ----------------------------------------------------------
				for key in [ 'embeddings', 'embedding_chunks', 'embedding_df',
						'embeddings_input_text' ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		# ------------------------------------------------------------------
		# TEXT METRICS (Render Above Buttons – Safe Append)
		# ------------------------------------------------------------------
		if st.session_state.get( 'embeddings_input_text' ):
			input_text: str = st.session_state.get( 'embeddings_input_text', '' ).strip( )
			
		if input_text:
			words = input_text.split( )
			total_words = len( words )
			unique_words = len( set( words ) )
			char_count = len( input_text )
			token_count = count_tokens( input_text )
			ttr = (unique_words / total_words) if total_words > 0 else 0.0
			
			col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns( 5, border=True )
			col_m1.metric( 'Tokens', token_count )
			col_m2.metric( 'Words', total_words )
			col_m3.metric( 'Unique Words', unique_words )
			col_m4.metric( 'TTR', f"{ttr:.3f}" )
			col_m5.metric( 'Characters', char_count )
			
			st.session_state[ 'embedding_metrics' ] = { 'tokens': token_count, 'words': total_words,
					'unique_words': unique_words, 'ttr': ttr, 'characters': char_count }
				

		# ------------------------------------------------------------------
		# EMBEDDING DATAFRAME (Dimension-Safe)
		# ------------------------------------------------------------------
		if 'embeddings' in st.session_state:
			embedding_vectors = st.session_state[ 'embeddings' ]
			
			# Normalize to 2D structure
			if isinstance( embedding_vectors, list ) and embedding_vectors:
				if isinstance( embedding_vectors[ 0 ], float ):
					embedding_vectors = [ embedding_vectors ]
				
				df_embedding = pd.DataFrame( embedding_vectors,
					columns=[ f"dim_{i}" for i in range( len( embedding_vectors[ 0 ] ) ) ] )
				
				st.data_editor( df_embedding, use_container_width=True, hide_index=True,
					key='embedding_vectors' )
					
# ======================================================================================
# VECTOR MODE
# ======================================================================================
elif mode == 'Vector Stores':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	stores_model = st.session_state.get( 'stores_model', None )
	stores_format = st.session_state.get( 'stores_response_format', None )
	stores_top_percent = st.session_state.get( 'stores_top_percent', None )
	stores_frequency = st.session_state.get( 'stores_frequency_penalty', None )
	stores_presense = st.session_state.get( 'stores_presense_penalty', None )
	stores_number = st.session_state.get( 'stores_number', None )
	stores_temperature = st.session_state.get( 'stores_temperature', None )
	stores_stream = st.session_state.get( 'stores_stream', None )
	stores_store = st.session_state.get( 'stores_store', None )
	stores_input = st.session_state.get( 'stores_input', None )
	stores_reasoning = st.session_state.get( 'stores_reasoning', None )
	vectorstores_choice = st.session_state.get( 'stores_tool_choice', None )
	stores_messages = st.session_state.get( 'stores_messages', None )
	stores_background = st.session_state.get( 'stores_background', None )
	vector = None
	collector  = None
	searcher = None
	if provider_name == 'Grok':
		st.subheader( '📚 Collections', help=cfg.VECTORSTORES_API )
		st.divider( )
	elif provider_name == 'Gemini':
		st.subheader( '📦 File Search Stores', help=cfg.VECTORSTORES_API )
		st.divider( )
	elif provider_name == 'GPT':
		st.subheader( '📦 Vector Stores', help=cfg.VECTORSTORES_API )
		st.divider( )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	if provider_name == 'Grok':
		provider_module = get_provider_module( )
		collector = provider_module.VectorStores( )
		
		# --------------------------------------------------------------
		# Local mapping (if maintained by wrapper)
		# --------------------------------------------------------------
		vs_map = getattr( collector, 'collections', None )
		if vs_map and isinstance( vs_map, dict ):
			st.markdown( '**Known Collections (local mapping)**' )
			for name, vid in vs_map.items( ):
				st.write( f"- **{name}** — `{vid}`" )
			st.markdown( "---" )
		
		# --------------------------------------------------------------
		# Create Collection
		# --------------------------------------------------------------
		with st.expander( 'Create:', expanded=True ):
			new_store_name = st.text_input( 'Enter Collection Name' )
			if st.button( '➕ Create Collection', key='create_collection' ):
				if not new_store_name:
					st.warning( 'Enter a Collection Name.' )
				else:
					try:
						if hasattr( collector, "create" ):
							res = provider_module.create( new_store_name )
							st.success( f"Create call submitted for '{new_store_name}'." )
						else:
							st.warning( 'create() not available on Grok provider.' )
					except Exception as exc:
						st.error( f'Create collection failed: {exc}' )
		
		# --------------------------------------------------------------
		# Discover collections ( API fallback )
		# --------------------------------------------------------------
		options: List[ tuple ] = [ ]
		if vs_map and isinstance( vs_map, dict ):
			options = list( vs_map.items( ) )
		
		if not options:
			try:
				client = getattr( collector, 'client', None )
				if ( client and hasattr( client, 'collections' )
						and hasattr( client.collections, 'list' ) ):
					api_list = client.collections.list( )
					temp: List[ tuple ] = [ ]
					for item in getattr( api_list, 'data', [ ] ) or api_list:
						nm = getattr( item, 'name', None ) or (
								item.get( 'name' ) if isinstance( item, dict ) else None )
						vid = getattr( item, 'id', None ) or (
								item.get( 'id' ) if isinstance( item, dict ) else None )
						if nm and vid:
							temp.append( (nm, vid) )
					if temp:
						options = temp
			except Exception:
				options = [ ]
		
		# --------------------------------------------------------------
		# Select / Retrieve / Delete
		# --------------------------------------------------------------
		if options:
			names = [ f"{n} — {i}" for n, i in options ]
			sel = st.selectbox( 'Select a Collection', options=names, key='sel_collection' )
			
			sel_id: Optional[ str ] = None
			for n, i in options:
				if f"{n} — {i}" == sel:
					sel_id = i
					break
			
			c1, c2 = st.columns( [ 1,  1 ] )
			with c1:
				if st.button( 'Retrieve Collection', key='retrieve_collection' ):
					if not sel_id:
						st.warning( 'No Collection Selected.' )
					else:
						try:
							client = getattr( collector, 'client', None )
							if ( client and hasattr( client, 'collections' )
									and hasattr( client.collections, 'retrieve' ) ):
								vs = client.collections.retrieve( collection_id=sel_id )
								st.json( vs.__dict__ if hasattr( vs, '__dict__' ) else vs )
							else:
								st.warning( 'collections.retrieve() not available.' )
						except Exception as exc:
							st.error( f'retrieve() failed: {exc}' )
			
			with c2:
				if st.button( '❌ Delete Collection',  key='delete_collection' ):
					if not sel_id:
						st.warning( 'No collection selected.' )
					else:
						try:
							client = getattr( collector, 'client', None )
							if ( client and hasattr( client, 'collections' )
									and hasattr( client.collections, 'delete' ) ):
								res = client.collections.delete( collection_id=sel_id )
								st.success( f'Delete returned: {res}' )
							else:
								st.warning( 'collections.delete() not available.' )
						except Exception as exc:
							st.error( f'Delete failed: {exc}' )
				else:
					st.info(
						'No collections discovered. Create one or confirm '
						'collections exist for this account.' )
						
	elif provider_name == 'Gemini':
		provider_module = get_provider_module( )
		searcher = provider_module.VectorStores( )
		
		# --------------------------------------------------------------
		# Local mapping (if maintained by wrapper)
		# --------------------------------------------------------------
		vs_map = getattr( searcher, 'collections', None )
		if vs_map and isinstance( vs_map, dict ):
			st.markdown( 'Local File Search Stores' )
			for name, vid in vs_map.items( ):
				st.write( f'- **{name}** — {vid}' )
			st.divider( )
		
		# --------------------------------------------------------------
		# Create File Search Store
		# --------------------------------------------------------------
		with st.expander( 'Create:', expanded=True ):
			new_store_name = st.text_input( 'New File Search Store name' )
			if st.button( '➕ Create File Search Store' ):
				if not new_store_name:
					st.warning( 'Enter a File Search Store Name.' )
				else:
					try:
						if hasattr( provider_module, 'create' ):
							res = provider_module.create( new_store_name )
							st.success( f"Create call submitted for '{new_store_name}'." )
						else:
							st.warning( 'create() not available on Gemini provider.' )
					except Exception as exc:
						st.error( f'Create store failed: {exc}' )
		
		# --------------------------------------------------------------
		# Discover file search stores (local → API fallback)
		# --------------------------------------------------------------
		options: List[ tuple ] = [ ]
		if vs_map and isinstance( vs_map, dict ):
			options = list( vs_map.items( ) )
		
		if not options:
			try:
				client = getattr( searcher, 'client', None )
				if ( client and hasattr( client, 'file_search_stores' )
						and hasattr( client.file_search_stores, 'list' ) ):
					api_list = client.file_search_stores.list( )
					temp: List[ tuple ] = [ ]
					for item in getattr( api_list, 'data', [ ] ) or api_list:
						nm = getattr( item, 'name', None ) or (
								item.get( 'name' ) if isinstance( item, dict ) else None )
						vid = getattr( item, 'id', None ) or (
								item.get( 'id' ) if isinstance( item, dict ) else None )
						if nm and vid:
							temp.append( (nm, vid) )
					if temp:
						options = temp
			except Exception:
				options = [ ]
		
		# --------------------------------------------------------------
		# Select / Retrieve / Delete
		# --------------------------------------------------------------
		if options:
			names = [ f'{n} — {i}' for n, i in options ]
			sel = st.selectbox( 'Select a File Search Store', options=names )
			
			sel_id: Optional[ str ] = None
			for n, i in options:
				if f'{n} — {i}' == sel:
					sel_id = i
					break
			
			c1, c2 = st.columns( [ 1,  1 ] )
			
			with c1:
				if st.button( 'Retrieve File Store' ):
					if not sel_id:
						st.warning( 'No file search store selected.' )
					else:
						try:
							client = getattr( searcher, 'client', None )
							if ( client and hasattr( client, 'file_search_stores' )
									and hasattr( client.file_search_stores, 'retrieve' ) ):
								vs = client.file_search_stores.retrieve(
									file_search_store_id=sel_id )
								st.json( vs.__dict__ if hasattr( vs, '__dict__' ) else vs )
							else:
								st.warning( 'file_search_stores.retrieve() not available.' )
						except Exception as exc:
							st.error( f'retrieve() failed: {exc}' )
			
			with c2:
				if st.button( '❌ Delete File Store' ):
					if not sel_id:
						st.warning( 'No file search store selected.' )
					else:
						try:
							client = getattr( searcher, 'client', None )
							if ( client and hasattr( client, 'file_search_stores' )
									and hasattr( client.file_search_stores, 'delete' ) ):
								res = client.file_search_stores.delete(
									file_search_store_id=sel_id )
								st.success( f'Delete returned: {res}' )
							else:
								st.warning( 'file_search_stores.delete() not available.' )
						except Exception as exc:
							st.error( f'Delete failed: {exc}' )
				else:
					st.info( 'No file search stores discovered' )
			
	elif provider_name == 'GPT':
		provider_module = get_provider_module( )
		vector = provider_module.VectorStores( )
		
		# --------------------------------------------------------------
		# Local mapping
		# --------------------------------------------------------------
		vs_map = getattr( vector, 'collections', None )
		st.caption( 'Store Management')
		# ------------------------------------------------------------------
		# Main Chat UI
		# ------------------------------------------------------------------
		left, center, right = st.columns( [ 0.025, 0.95, 0.055 ] )
		with center:
			stores_left, stores_right = st.columns( [ 0.50, 0.50 ], border=True )
			with stores_left:
				# --------------------------------------------------------------
				# Expander - Create Vector Store
				# --------------------------------------------------------------
				with st.expander( 'Create:', expanded=True ):
					new_store_name = st.text_input( 'New Vector Store name', key='store_name' )
					if st.button( '➕ Create Store', key='create_store' ):
						if not new_store_name:
							st.warning( 'Enter a Vector Store Name.' )
						else:
							try:
								if hasattr( vector, 'create' ):
									res = vector.create( new_store_name )
									st.success( f"Create call submitted for '{new_store_name}'." )
								else:
									st.warning( 'create() not available on VectorStores wrapper.' )
							except Exception as exc:
								st.error( f'Create store failed: {exc}' )
			
			with stores_right:
				# --------------------------------------------------------------
				# Discover vector stores
				# --------------------------------------------------------------
				with st.expander( 'Retreive:', expanded=True ):
					options: List[ tuple ] = [ ]
					if vs_map and isinstance( vs_map, dict ):
						options = list( vs_map.items( ) )
					
					if not options:
						try:
							openai_client = st.session_state.get( 'openai_client' )
							if ( openai_client and hasattr( openai_client, 'vector_stores' )
									and hasattr( openai_client.vector_stores, 'list' ) ):
								api_list = openai_client.vector_stores.list( )
								temp: List[ tuple ] = [ ]
								for item in getattr( api_list, 'data', [ ] ) or api_list:
									nm = getattr( item, 'name', None ) or (
											item.get( 'name' ) if isinstance( item, dict ) else None )
									vid = getattr( item, 'id', None ) or (
											item.get( 'id' ) if isinstance( item, dict ) else None )
									if nm and vid:
										temp.append( (nm, vid) )
								if temp:
									options = temp
						except Exception:
							options = [ ]
						
					# --------------------------------------------------------------
					# Select / Retrieve / Delete
					# --------------------------------------------------------------
					if options:
						names = [ f'{n} — {i}' for n, i in options ]
						sel = st.selectbox( 'Select Vector Store', options=names,
							key='select_vectorstore' )
						
						sel_id: Optional[ str ] = None
						for n, i in options:
							if f'{n} — {i}' == sel:
								sel_id = i
								break
						
						c1, c2 = st.columns( [ 1, 1 ] )
						
						with c1:
							if st.button( 'Retrieve Store', key='retrieve_store' ):
								if not sel_id:
									st.warning( 'No vector store selected.' )
								else:
									try:
										openai_client = st.session_state[ 'openai_client' ]
										vs = openai_client.vector_stores.retrieve( vector_store_id=sel_id )
										st.write( 'Name:', vs.name)
										st.write( 'Files:', vs.file_counts )
										st.write( 'Size (MB):', round( vs.usage_bytes / 1_048_576, 2 ) )
									except Exception as exc:
										st.error( f'retrieve() failed: {exc}' )
						
						with c2:
							if st.button( '❌ Delete', key='delete_store' ):
								if not sel_id:
									st.warning( 'No vector store selected.' )
								else:
									try:
										openai_client = st.session_state.get( 'openai_client' )
										if openai_client and hasattr( openai_client.vector_stores, 'delete' ):
											res = openai_client.vector_stores.delete( vector_store_id=sel_id )
											st.success( f'Delete returned: {res}' )
										else:
											st.warning( 'vector_stores.delete() not available.' )
									except Exception as exc:
										st.error( f'Delete failed: {exc}' )
					else:
						st.info( 'No vector stores discovered' )
						
# ======================================================================================
# DOCUMENTS MODE
# ======================================================================================
elif mode == 'Document Q&A':
	st.subheader( '📚 Document Q & A', help=cfg.DOCUMENT_Q_AND_A )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	files = st.session_state.get( 'docqna_files' )
	uploaded = st.session_state.get( 'docqna_uploaded' )
	docqna_messages = st.session_state.get( 'docqna_messages' )
	doc_active_docs = st.session_state.get( 'docqna_active_docs' )
	doc_source = st.session_state.get( 'docqna_source' )
	doc_multi_mode = st.session_state.get( 'docqna_multi_mode' )
	
	# ------------------------------------------------------------------
	#  DOCQA SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'do_clear_instructions' ):
		st.session_state[ 'docqna_system_instructions' ] = ''
		st.session_state[ 'clear_docqa_instructions' ] = False
		st.session_state[ 'do_clear_instructions' ] = False
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — DocQA Inference Parameters
		# ------------------------------------------------------------------
		with st.expander( '🧠 Inference Settings', expanded=True, width='stretch' ):
			stores_c1, stores_c2, stores_c3, stores_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
				border=True, gap='medium' )
			
			with stores_c1:
				set_text_top_p = st.slider( 'Top-P', 0.0, 1.0,
					float( st.session_state.get( 'top_p', 1.0 ) ), 0.01,
					help=cfg.TOP_P, key='text_top_percent' )
				text_top_percent = st.session_state[ 'text_top_percent' ]
			
			with stores_c2:
				set_text_freq = st.slider( 'Frequency Penalty', -2.0, 2.0,
					float( st.session_state.get( 'text_frequency_penalty', 0.0 ) ),
					0.01, help=cfg.FREQUENCY_PENALTY )
				text_fequency = st.session_state[ 'text_frequency_penalty' ]
			
			with stores_c3:
				set_text_presense = st.slider( 'Presence Penalty', -2.0, 2.0,
					float( st.session_state.get( 'text_presense_penalty', 0.0 ) ),
					0.01, help=cfg.PRESENCE_PENALTY )
				text_presense = st.session_state[ 'text_presense_penalty' ]
			
			with stores_c4:
				set_text_temperature = st.slider( 'Temperature', 0.0, 1.0,
					float( st.session_state.get( 'text_temperature', 0.7 ) ), 0.01,
					help=cfg.TEMPERATURE )
				text_temperature = st.session_state[ 'text_temperature' ]
			
			if st.button( 'Reset', key='docqa_inference_reset', width='stretch' ):
				# ----------------------------------------------------------
				# Remove DocQA Inference session keys
				# ----------------------------------------------------------
				for key in [ 'docqa_temperature', 'docqa_top_p', 'docqa_max_tokens',
						'docqa_top_k', 'docqa_score_threshold', ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — DocQA System Instructions
		# ------------------------------------------------------------------
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			in_left, in_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			with in_left:
				st.text_area( 'Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='docqna_system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'docqna_system_instructions' ] = text
			
			with in_right:
				st.selectbox( 'Select Template', prompt_names,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'docqna_system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( 'Clear Instructions', width='stretch', on_click=_on_clear )
		
		doc_left, doc_right = st.columns( [ 0.2, 0.8 ], border=True )
		with doc_left:
			uploaded = st.file_uploader( 'Upload', type=[ 'pdf', 'txt', 'md', 'docx' ],
				accept_multiple_files=False, label_visibility='visible' )
			
			if uploaded is not None:
				st.session_state.docqna_active_docs = [ uploaded.name ]
				st.session_state.doc_bytes = { uploaded.name: uploaded.getvalue( ) }
				st.success( f'{uploaded.name} has been loaded!' )
			else:
				st.info( 'Load a document.' )
			
			unload = st.button( label='Unload Document', width='stretch' )
			if unload:
				uploaded = None
				st.session_state.docqna_active_docs = None
		
		with doc_right:
			if st.session_state.get( 'docqna_active_docs' ):
				name = st.session_state.docqna_active_docs[ 0 ]
				file_bytes = st.session_state.doc_bytes.get( name )
				if file_bytes:
					st.pdf( file_bytes, height=420 )
		
		for msg in st.session_state.docqna_messages:
			with st.chat_message( msg[ 'role' ] ):
				st.markdown( msg[ 'content' ] )
		
		if prompt := st.chat_input( 'Ask a question about the document' ):
			st.session_state.docqna_messages.append( { 'role': 'user', 'content': prompt } )
			response = route_document_query( prompt )
			st.session_state.docqna_messages.append( { 'role': 'assistant', 'content': response } )
			st.rerun( )

# ======================================================================================
# FILES API MODE
# ======================================================================================
elif mode == 'Files':
	st.subheader( '📁 Files API', help=cfg.FILES_API )
	st.divider( )
	purpose = st.session_state.get( 'files_purpose' )
	file_type = st.session_state.get( 'files_type' )
	file_id = st.session_state.get( 'files_id' )
	file_url = st.session_state.get( 'files_url' )
	try:
		chat  # type: ignore
	except NameError:
		provider_module = get_provider_module( )
		files = get_provider_module( ).Files( )
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		list_method = None
		if hasattr( files, 'list' ):
			list_method = getattr( files, 'list' )
		
		uploaded_file = st.file_uploader( 'Upload file (server-side via Files API)',
			type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', ], )
		if uploaded_file:
			tmp_path = save_temp( uploaded_file )
			upload_fn = None
			for name in ('upload_file', 'upload', 'files_upload'):
				if hasattr( files, name ):
					upload_fn = getattr( files, name )
					break
			if not upload_fn:
				st.warning( 'No upload function found on chat object.' )
			else:
				with st.spinner( 'Uploading to Files API...' ):
					try:
						fid = upload_fn( tmp_path )
						st.success( f'Uploaded; file id: {fid}' )
					except Exception as exc:
						st.error( f"Upload failed: {exc}" )
	
		if st.button( 'List Files' ):
			try:
				files_resp = list_method( )
				rows = [ ]
				files_list = ( files_resp.data if hasattr( files_resp, 'data' ) else files_resp
						if isinstance( files_resp, list ) else [ ] )
				
				for f in files_list:
					rows.append( { 'id': str( getattr( f, 'id', "" ) ),
							'filename': str( getattr( f, 'filename', "" ) ),
							'files_purpose': str( getattr( f, 'files_purpose', "" ) ), } )
				
				st.session_state.files_table = rows
			
			except Exception as exc:
				st.session_state.files_table = None
				st.error( f'List files failed: {exc}' )
		
			if 'files_list' in locals( ) and files_list:
					file_ids = [ r.get( 'filename' ) if isinstance( r, dict )
					             else getattr( r, 'id', None ) for r in files_list ]
					sel = st.selectbox( 'Select File to Delete', options=file_ids )
					if st.button( 'Delete File' ):
						del_fn = None
						for name in ( 'delete_file', 'delete', 'files_delete' ):
							if hasattr( files, name ):
								del_fn = getattr( files, name )
								break
						if not del_fn:
							st.warning( 'No delete function found on chat object.' )
						else:
							with st.spinner( 'Deleting file...' ):
								try:
									res = del_fn( sel )
									st.success( f'Delete result: {res}' )
								except Exception as exc:
									st.error( f'Delete failed: {exc}' )

# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == 'Prompt Engineering':
	st.subheader( '📝 Prompt Engineering', help=cfg.PROMPT_ENGINEERING )
	st.divider( )
	import sqlite3
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	st.session_state.setdefault( 'pe_cascade_enabled', False )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.checkbox( 'Cascade selection into System Instructions', key='pe_cascade_enabled' )
		
		# ------------------------------------------------------------------
		# Session state
		# ------------------------------------------------------------------
		st.session_state.setdefault( 'pe_page', 1 )
		st.session_state.setdefault( 'pe_search', '' )
		st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
		st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
		st.session_state.setdefault( 'pe_selected_id', None )
		st.session_state.setdefault( 'pe_caption', '' )
		st.session_state.setdefault( 'pe_name', '' )
		st.session_state.setdefault( 'pe_text', '' )
		st.session_state.setdefault( 'pe_version', '' )
		st.session_state.setdefault( 'pe_id', 0 )
		
		# ------------------------------------------------------------------
		# DB helpers
		# ------------------------------------------------------------------
		def get_conn( ):
			return sqlite3.connect( cfg.DB_PATH )
		
		def reset_selection( ):
			st.session_state.pe_selected_id = None
			st.session_state.pe_caption = ''
			st.session_state.pe_name = ''
			st.session_state.pe_text = ''
			st.session_state.pe_version = ''
			st.session_state.pe_id = 0
		
		def load_prompt( pid: int ) -> None:
			with get_conn( ) as conn:
				_select = f"SELECT Caption, Name, Text, Version, ID FROM {TABLE} WHERE PromptsId=?"
				cur = conn.execute( _select, (pid,), )
				row = cur.fetchone( )
				if not row:
					return
				st.session_state.pe_caption = row[ 0 ]
				st.session_state.pe_name = row[ 1 ]
				st.session_state.pe_text = row[ 2 ]
				st.session_state.pe_version = row[ 3 ]
				st.session_state.pe_id = row[ 4 ]
		
		# ------------------------------------------------------------------
		# Filters
		# ------------------------------------------------------------------
		c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
		
		with c1:
			st.text_input( 'Search (Name/Text contains)', key='pe_search' )
		
		with c2:
			st.selectbox( 'Sort by', [ 'PromptsId', 'Caption', 'Name', 'Text', 'Version', 'ID' ],
				key='pe_sort_col', )
		
		with c3:
			st.selectbox( 'Direction', [ 'ASC', 'DESC' ], key='pe_sort_dir' )
		
		with c4:
			st.markdown(
				"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
				unsafe_allow_html=True, )
			
			a1, a2, a3 = st.columns( [ 2, 1, 1 ] )
			
			with a1:
				jump_id = st.number_input( 'Go to ID', min_value=1,
					step=1, label_visibility='collapsed', )
			
			with a2:
				if st.button( 'Go' ):
					st.session_state.pe_selected_id = int( jump_id )
					load_prompt( int( jump_id ) )
			
			with a3:
				st.button( 'Clear', on_click=reset_selection )
		
		# ------------------------------------------------------------------
		# Load prompt table
		# ------------------------------------------------------------------
		where = ""
		params = [ ]
		if st.session_state.pe_search:
			where = 'WHERE Name LIKE ? OR Text LIKE ?'
			s = f"%{st.session_state.pe_search}%"
			params.extend( [ s, s ] )
		
		offset = (st.session_state.pe_page - 1) * PAGE_SIZE
		query = f"""
	        SELECT PromptsId, Caption, Name, Text, Version, ID
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
					'Caption': r[ 1 ],
					'Name': r[ 2 ],
					'Text': r[ 3 ],
					'Version': r[ 4 ],
					'ID': r[ 5 ],
			} )
		
		edited = st.data_editor( table_rows, hide_index=True, use_container_width=True,
			key="prompt_table", )
		
		# ------------------------------------------------------------------
		# SELECTION PROCESSING (must run BEFORE widgets below)
		# ------------------------------------------------------------------
		selected = [ r for r in edited if isinstance( r, dict ) and r.get( 'Selected' ) ]
		if len( selected ) == 1:
			pid = int( selected[ 0 ][ 'PromptsId' ] )
			if pid != st.session_state.pe_selected_id:
				st.session_state.pe_selected_id = pid
				load_prompt( pid )
		
		elif len( selected ) == 0:
			reset_selection( )
		
		elif len( selected ) > 1:
			st.warning( 'Select exactly one prompt row.' )
		
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
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

		
		# ------------------------------------------------------------------
		# Edit Prompt
		# ------------------------------------------------------------------
		with st.expander( "🖊️ Edit Prompt", expanded=False ):
			st.text_input( "PromptsId", value=st.session_state.pe_selected_id or "",
				disabled=True, )
			st.text_input( 'Name', key='pe_name' )
	
			st.text_area( 'Text', key='pe_text', height=260 )
			st.text_input( 'Version', key='pe_version' )
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				if st.button( '💾 Save Changes'
				if st.session_state.pe_selected_id
				else '➕ Create Prompt' ):
					with get_conn( ) as conn:
						if st.session_state.pe_selected_id:
							conn.execute(
								f"""
	                            UPDATE {TABLE}
	                            SET Caption=?, Name=?, Text=?, Version=?, ID=?
	                            WHERE PromptsId=?
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id,
										st.session_state.pe_selected_id
								), )
						else:
							conn.execute(
								f"""
	                            INSERT INTO {TABLE} (Caption, Name, Text, Version, ID)
	                            VALUES (?, ?, ?, ? , ?)
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id
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
							(st.session_state.pe_selected_id,), )
						conn.commit( )
					reset_selection( )
					st.success( 'Deleted.' )
			
			with c3:
				st.button( '🧹 Clear Selection', on_click=reset_selection )
				
# ==============================================================================
# EXPORT MODE
# ==============================================================================
elif mode == 'Data Export':
	st.subheader( '📭  Export' )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.divider( )
		
		# -----------------------------------
		# Prompt export (System Instructions)
		st.caption( 'System Prompt' )
		export_format = st.radio( 'Export Format', options=[ 'XML-Delimited', 'Markdown' ],
			horizontal=True, help='Choose how system instructions should be exported.' )
		prompt_text: str = st.session_state.get( 'system_prompt', '' )
		if export_format == 'Markdown':
			try:
				export_text: str = convert_xml( prompt_text )
				export_filename: str = 'Buddy_Instructions.md'
			except Exception as exc:
				st.error( f'Markdown conversion failed: {exc}' )
				export_text = ''
				export_filename = ''
		else:
			export_text = prompt_text
			export_filename = 'Buddy_System_Instructions.xml'
		
		st.download_button( label='Download System Instructions', data=export_text,
			file_name=export_filename, mime='text/plain', disabled=not bool( export_text.strip( ) ) )
		
		# -----------------------------
		# Existing chat history export
		st.divider( )
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

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	st.subheader( "🏛️ Data Management", help=cfg.DATA_MANAGEMENT )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ "📥 Import", "🗂 Browse", "💉 CRUD", "📊 Explore", "🔎 Filter",
				"🧮 Aggregate", "📈 Visualize", "⚙ Admin", "🧠 SQL" ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		else:
			table = st.selectbox( "Table", tables )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany(
								insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( )
							)
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
			
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				st.dataframe( df, use_container_width=True )
			else:
				st.info( 'No tables available.' )
	
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.subheader( 'Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[ column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[
							column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[ column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.subheader( 'Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.subheader( 'Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.dataframe( df_page, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				st.dataframe( df, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					agg = st.selectbox( 'Function', [ 'SUM',  'AVG', 'COUNT' ] )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
			
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					fig = px.histogram( df, x=col )
					st.plotly_chart( fig, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					st.dataframe( profile_df, use_container_width=True )
					
			st.subheader( 'Drop Table' )
	
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
						'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
					
			st.divider( )
			
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.dataframe( schema_df, use_container_width=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame(
						indexes,
						columns=[ 'seq', 'name',  'unique',  'origin', 'partial' ]
					)
					st.markdown( "### Indexes" )
					st.dataframe( idx_df, use_container_width=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER',  'REAL',  'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						dm_rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						dm_rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
						
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ======================================================================================
# FOOTER — SECTION
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

# ---- Fixed Container
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
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #35618c;
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

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
_mode_to_model_key = \
{
	'Text': 'text_model',
	'Images': 'image_model',
	'Audio': 'audio_model',
	'Embeddings': 'embed_model',
}

provider_val = st.session_state.get( 'provider', '—' )
mode_val = mode or '—'
active_model = st.session_state.get( _mode_to_model_key.get( mode, "" ), None )
right_parts = [ ]
if active_model is not None:
	right_parts.append( f'Model: {active_model}' )

# ---- Rendered Variables
if mode == 'Text':
	temperature = st.session_state.get( 'text_temperature' )
	top_p = st.session_state.get( 'text_top_percent' )
	freq = st.session_state.get( 'text_frequency_penalty' )
	presence = st.session_state.get( 'text_presense_penalty' )
	number = st.session_state.get( 'text_number' )
	stream = st.session_state.get( 'text_stream' )
	parallel_tools = st.session_state.get( 'text_parallel_tools' )
	max_calls = st.session_state.get( 'text_max_tools' )
	store = st.session_state.get( 'text_store' )
	tools = st.session_state.get( 'text_tools' )
	include = st.session_state.get( 'text_include' )
	domains = st.session_state.get( 'text_domains' )
	input_mode = st.session_state.get( 'text_input' )
	tool_choice = st.session_state.get( 'text_tool_choice' )
	background = st.session_state.get( 'text_background' )
	messages = st.session_state.get( 'text_messages' )
	max_tokens = st.session_state.get( 'text_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.0%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.0%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.0%}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.0%}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )

elif mode == 'Images':
	image_mode = st.session_state.get( 'image_mode' )
	image_size = st.session_state.get( 'image_size' )
	image_aspect = st.session_state.get( 'image_aspect' )
	image_style = st.session_state.get( 'image_style' )
	image_backcolor = st.session_state.get( 'image_backcolor' )
	image_quality = st.session_state.get( 'image_quality' )
	image_fmt = st.session_state.get( 'image_format' )
	image_reasoning = st.session_state.get( 'image_reasoning' )
	image_detail = st.session_state.get( 'image_detail' )
	image_number = st.session_state.get( 'image_number' )
	image_stream = st.session_state.get( 'image_stream' )
	image_store = st.session_state.get( 'image_store' )
	image_background = st.session_state.get( 'image_background' )
	image_include = st.session_state.get( 'image_include' )
	image_parallel_tools = st.session_state.get( 'image_parallel_tools' )
	image_max_calls = st.session_state.get( 'text_max_tools' )
	image_tools = st.session_state.get( 'image_tools' )
	
	if image_aspect is not None:
		right_parts.append( f'Aspect: {image_aspect}' )
	elif image_size is not None:
		right_parts.append( f'Size: {image_size}' )
	
	if image_mode is not None:
		right_parts.append( f'Mode: {image_mode}' )
	if image_reasoning is not None:
		right_parts.append( f'Reasoning: {image_reasoning}' )
	if image_style is not None:
		right_parts.append( f'Style: {image_style}' )
	if image_quality is not None:
		right_parts.append( f'Quality: {image_quality}' )
	if image_backcolor is not None:
		right_parts.append( f'Backcolor: {image_backcolor}' )
	if image_fmt is not None:
		right_parts.append( f'Format: {image_fmt}' )
	if image_detail is not None:
		right_parts.append( f'Detail: {image_detail}' )
	
	if image_number is not None:
		right_parts.append( f'N: {image_number}' )
	if image_parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if image_max_calls is not None:
		right_parts.append( f'Max Calls: {image_max_calls}' )
	if image_tools:
		right_parts.append( f'Tools: {len(image_tools )}' )
	if image_include:
		right_parts.append( 'Include: On' )
	if image_stream:
		right_parts.append( 'Stream: On' )
	if image_store:
		right_parts.append( 'Store: On' )
	if image_background:
		right_parts.append( 'Background: On' )

elif mode == 'Audio':
	task = st.session_state.get( 'audio_task' )
	fmt = st.session_state.get( 'audio_response_format' )
	top_p = st.session_state.get( 'audio_top_percent' )
	freq = st.session_state.get( 'audio_frequency_penalty' )
	presence = st.session_state.get( 'audio_presense_penalty' )
	number = st.session_state.get( 'audio_number' )
	temperature = st.session_state.get( 'audio_temperature' )
	stream = st.session_state.get( 'audio_stream' )
	store = st.session_state.get( 'audio_store' )
	input_mode = st.session_state.get( 'audio_input' )
	reasoning = st.session_state.get( 'audio_reasoning' )
	tool_choice = st.session_state.get( 'audio_tool_choice' )
	messages = st.session_state.get( 'audio_messages' )
	background = st.session_state.get( 'audio_background' )
	audio_file = st.session_state.get( 'audio_file' )
	rate = st.session_state.get( 'audio_rate', 16000 )
	start = st.session_state.get( 'audio_start', 0.0 )
	end = st.session_state.get( 'audio_end', 0.0 )
	loop = st.session_state.get( 'audio_loop', False )
	autoplay = st.session_state.get( 'auto_play', False )
	voice = st.session_state.get( 'voice', None )
	
	if task is not None:
		right_parts.append( f'Task: {task}' )
	if fmt is not None:
		right_parts.append( f'Format: {fmt}' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if store:
		right_parts.append( 'Store: On' )
	if reasoning:
		right_parts.append( 'Reasoning: On' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: {tool_choice}' )
	if messages:
		right_parts.append( 'Messages: Set' )
	if background:
		right_parts.append( 'Background: On' )
	
	if voice:
		right_parts.append( f'Voice: {voice}' )
	if rate is not None:
		right_parts.append( f'Rate: {rate}' )
	if (start or end) and end >= start:
		right_parts.append( f'Trim: {start}s–{end}s' )
	if loop:
		right_parts.append( 'Loop: On' )
	if autoplay:
		right_parts.append( 'Autoplay: On' )
	if audio_file is not None:
		right_parts.append( 'File: Set' )

elif mode == 'Embeddings':
	model = st.session_state.get( 'embedding_model' )
	dimensions = st.session_state.get( 'embeddings_dimensions' )
	encoding = st.session_state.get( 'embeddings_encoding_format' )
	input_data = st.session_state.get( 'embedding_text_input' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if dimensions is not None:
		right_parts.append( f'Dim: {dimensions}' )
	
	if encoding is not None:
		right_parts.append( f'Format: {encoding}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )

elif mode == 'Files':
	purpose = st.session_state.get( 'files_purpose' )
	file_type = st.session_state.get( 'files_type' )
	file_id = st.session_state.get( 'files_id' )
	file_url = st.session_state.get( 'files_url' )
	
	if purpose is not None:
		right_parts.append( f'Purpose: {purpose}' )
	
	if file_type is not None:
		right_parts.append( f'Type: {file_type}' )
	
	if file_id is not None:
		right_parts.append( f'File ID: {file_id}' )
	
	if file_url is not None:
		right_parts.append( 'URL: Set' )

elif mode == 'VectorStores':
	model = st.session_state.get( 'stores_model' )
	fmt = st.session_state.get( 'stores_response_format' )
	temperature = st.session_state.get( 'stores_temperature' )
	top_p = st.session_state.get( 'stores_top_percent' )
	freq = st.session_state.get( 'stores_frequency_penalty' )
	presence = st.session_state.get( 'stores_presense_penalty' )
	number = st.session_state.get( 'stores_number' )
	stream = st.session_state.get( 'stores_stream' )
	store = st.session_state.get( 'stores_store' )
	input_data = st.session_state.get( 'stores_input' )
	reasoning = st.session_state.get( 'stores_reasoning' )
	tool_choice = st.session_state.get( 'stores_tool_choice' )
	messages = st.session_state.get( 'stores_messages' )
	background = st.session_state.get( 'stores_background' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if fmt is not None:
		right_parts.append( f'Format: {fmt}' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )
	
	if freq is not None:
		right_parts.append( f'Freq: {freq}' )
	
	if presence is not None:
		right_parts.append( f'Presence: {presence}' )
	
	if number is not None:
		right_parts.append( f'N: {number}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	
	if store:
		right_parts.append( 'Store: On' )
	
	if reasoning is not None:
		right_parts.append( f'Reasoning: {reasoning}' )
	
	if tool_choice is not None:
		right_parts.append( f'Tool Choice: {tool_choice}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )
	
	if messages:
		right_parts.append( 'Messages: Set' )
	
	if background:
		right_parts.append( 'Background: On' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
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