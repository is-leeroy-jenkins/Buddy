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
import typing_extensions
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
		
	for k in ( 'audio_system_instructions',
				'image_system_instructions',
				'text_system_instructions', ):
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

DM_DB_PATH = os.path.join( "stores", "sqlite", "Data.db" )
os.makedirs( os.path.dirname( DM_DB_PATH ), exist_ok=True )

# ==============================================================================
# DATABASE CORE
# ==============================================================================
def dm_conn( ) -> sqlite3.Connection:
	return sqlite3.connect( DM_DB_PATH )

def dm_tables( ) -> List[ str ]:
	with dm_conn( ) as conn:
		rows = conn.execute(
			"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		).fetchall( )
		return [ r[ 0 ] for r in rows ]

def dm_schema( table: str ) -> List[ Tuple ]:
	with dm_conn( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def dm_read( table: str, limit: int = None, offset: int = 0 ) -> pd.DataFrame:
	query = f'SELECT rowid, * FROM "{table}"'
	if limit:
		query += f" LIMIT {limit} OFFSET {offset}"
	with dm_conn( ) as conn:
		return pd.read_sql_query( query, conn )

def dm_drop_table( table: str ):
	with dm_conn( ) as conn:
		conn.execute( f'DROP TABLE "{table}";' )
		conn.commit( )

def dm_create_index( table: str, column: str ) -> None:
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
	tables = dm_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = dm_schema( table )
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
	
	with dm_conn( ) as conn:
		conn.execute( sql )
		conn.commit( )

def dm_apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
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

def dm_aggregation( df: pd.DataFrame ):
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

def dm_visualization( df: pd.DataFrame ):
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
		sql_type = dm_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}')
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with dm_conn( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def dm_insert_df( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with dm_conn( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def dm_sqlite_type( dtype ) -> str:
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

def dm_create_custom_table( table_name: str, columns: list ) -> None:
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
	
	with dm_conn( ) as conn:
		conn.execute( sql )
		conn.commit( )

# ==============================================================================
# Page Setup / Configuration
# ==============================================================================
openai_client = OpenAI( )
st.session_state[ 'openai_client' ] = openai_client

AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BUDDY, }

st.set_page_config( page_title=cfg.APP_TITLE, layout='wide',
	page_icon=cfg.FAVICON, initial_sidebar_state='collapsed' )

st.caption( cfg.APP_SUBTITLE )

inject_response_css( )

init_state( )


# ======================================================================================
# Session State — initialize per-mode model keys and token counters
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

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0, }

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0, }

if 'files' not in st.session_state:
	st.session_state.files: List[ str ] = [ ]
	
#----------MODELS--------------------------------
if 'model' not in st.session_state:
	st.session_state.model = None

if 'text_model' not in st.session_state:
	st.session_state[ 'text_model' ] = None
	
if 'image_model' not in st.session_state:
	st.session_state[ 'image_model' ] = None
	
if 'audio_model' not in st.session_state:
	st.session_state[ 'audio_model' ] = None
	
if 'embeddings_model' not in st.session_state:
	st.session_state[ 'embeddings_model' ] = None

if 'tts_model' not in st.session_state:
	st.session_state[ 'tts_model' ] = None

if 'transcription_model' not in st.session_state:
	st.session_state[ 'transcription_model' ] = None

if 'translation_model' not in st.session_state:
	st.session_state[ 'translation_model' ] = None

# --------SYSTEM INSTRUCTIONS----------------------
if 'instructions' not in st.session_state:
	st.session_state[ 'instructions' ] = ''
	
if 'text_system_instructions' not in st.session_state:
	st.session_state[ 'text_system_instructions' ] = ''

if 'image_system_instructions' not in st.session_state:
	st.session_state[ 'image_system_instructions' ] = ''

if 'audio_system_instructions' not in st.session_state:
	st.session_state[ 'audio_system_instructions' ] = ''

if 'doc_instructions' not in st.session_state:
	st.session_state.doc_instructions = ''

#--------GENERATION CONTROLS--------------------
if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.8
	
if 'top_p' not in st.session_state:
	st.session_state[ 'top_p' ] = 1.0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 8064
	
if 'freq_penalty' not in st.session_state:
	st.session_state[ 'freq_penalty' ] = 0.0
	
if 'pres_penalty' not in st.session_state:
	st.session_state[ 'pres_penalty' ] = 0.0

if 'logprobs' not in st.session_state:
	st.session_state[ 'logprobs' ] = 0

if 'stop_sequences' not in st.session_state:
	st.session_state[ 'stop_sequences' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = 'auto'

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = 'low'

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = True

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'messages' not in st.session_state:
	st.session_state.messages: List[ Dict[ str, Any ] ] = [ ]

#-------AUDIO-API---------------------------
if 'audio_file' not in st.session_state:
	st.session_state[ 'audio_file' ] = None

if 'sample_rate' not in st.session_state:
	st.session_state[ 'sample_rate' ] = 16000

if 'language' not in st.session_state:
	st.session_state[ 'language' ] = None

if 'voice' not in st.session_state:
	st.session_state[ 'voice' ] = None

if 'audio_start' not in st.session_state:
	st.session_state[ 'audio_start' ] = 0.0

if 'audio_end' not in st.session_state:
	st.session_state[ 'audio_end' ] = 0.0

if 'audio_loop' not in st.session_state:
	st.session_state[ 'audio_loop' ] = False
	
if 'auto_play' not in st.session_state:
	st.session_state[ 'auto_play' ] = False

if 'audio_format' not in st.session_state:
	st.session_state[ 'audio_format' ] = 'audio/wav'

# ------- IMAGE API--------------------------
if 'image_size' not in st.session_state:
	st.session_state[ 'image_size' ] = None
	
if 'image_style' not in st.session_state:
	st.session_state[ 'image_style' ] = None

if 'image_detail' not in st.session_state:
	st.session_state[ 'image_detail' ] = None

if 'image_background' not in st.session_state:
	st.session_state[ 'image_background' ] = None

if 'image_quality' not in st.session_state:
	st.session_state[ 'image_quality' ] = None

if 'image_format' not in st.session_state:
	st.session_state[ 'image_format' ] = None

if 'image_url' not in st.session_state:
	st.session_state[ 'image_url' ] = None

if 'ascpect_ratio' not in st.session_state:
	st.session_state[ 'aspect_ratio' ] = None

# ------- FILES API--------------------------
if 'purpose' not in st.session_state:
	st.session_state[ 'purpose' ] = None

if 'file_type' not in st.session_state:
	st.session_state[ 'file_type' ] = None

if 'file_id' not in st.session_state:
	st.session_state[ 'file_id' ] = None

if 'file_url' not in st.session_state:
	st.session_state[ 'file_url' ] = None
	
# -------VECTOR STORES API-------------------
if 'vector_store_id' not in st.session_state:
	st.session_state[ 'vector_store_id' ] = None

#------- DOC Q&A  ---------------------------
if 'files' not in st.session_state:
	st.session_state[ 'files' ] = [ ]
	
if 'uploaded' not in st.session_state:
	st.session_state[ 'uploaded' ] = None

if 'doc_messages' not in st.session_state:
	st.session_state.doc_messages = [ ]

if 'doc_active_docs' not in st.session_state:
	st.session_state.doc_active_docs = [ ]

if 'doc_source' not in st.session_state:
	st.session_state.doc_source = None

if 'doc_multi_mode' not in st.session_state:
	st.session_state.doc_multi_mode = False

# ======================================================================================
#  PROVIDER
# ======================================================================================
def get_provider_module( ):
	provider = st.session_state.get( 'provider' )
	module_name = cfg.PROVIDERS.get( provider  )
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

#-------------DOC Q&A ----------------------
def route_document_query( prompt: str ) -> str:
	source = st.session_state.get( 'doc_source' )
	active_docs = st.session_state.get( 'doc_active_docs', [ ] )
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
		st.radio( 'Execution Mode', options=[ 'Standard', 'Guidance Only', 'Analysis Only' ],
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
			with st.chat_message( "user", avatar=cfg.ANALYST ):
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
							tools=[ { 'type': 'file_search', 'vector_store_ids': cfg.GPT_VECTORSTORE_IDS, },
									{ 'type': 'web_search', 'filters': { 'allowed_domains': cfg.GPT_WEB_DOMAINS, },
											'search_context_size': 'medium',
											'user_location': { 'type': 'approximate' },
									},
									{ 'type': 'code_interpreter',
									  'container': { 'type': 'auto', 'file_ids': cfg.GPT_FILE_IDS, },
									},
							],
							include=[ 'web_search_call.action.sources',
							          'code_interpreter_call.outputs', ],
							store=True, )
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
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	chat = provider_module.Chat( )
	
	# ------------------------------------------------------------------
	# Sidebar — Text Settings
	# ------------------------------------------------------------------
	with (st.sidebar):
		st.text( '⚙️ Text Settings' )
		
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		if st.session_state.get( 'do_clear_instructions', False ):
			st.session_state[ 'text_system_instructions' ] = ''
			st.session_state[ 'do_clear_instructions' ] = False
		
		with st.expander( '🧠 LLM Configuration', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Expander — LLM
			# ------------------------------------------------------------------
			with st.expander( 'Model Options', expanded=False, width='stretch' ):
					llm_one, llm_two, llm_three, llm_four, llm_five = st.columns( [ 0.2,
					                                                                0.2,
					                                                                0.2,
					                                                                0.2,
					                                                                0.2 ], border=True )
					
					with llm_one:
						text_model = st.selectbox( 'Select Model', chat.model_options,
							help='Required. Text Generation model used by the AI',
							index=(chat.model_options.index( st.session_state[ 'text_model' ] )
							       if st.session_state.get( 'text_model' ) in chat.model_options else 0), )
						st.session_state[ 'text_model' ] = text_model
					
					with llm_two:
						include = st.multiselect( 'Include:', options=chat.include_options,
							key='chat_include', help=cfg.INCLUDE )
						chat.include = include
						st.session_state[ 'include' ] = include
					
					with llm_three:
						tool_choice = st.multiselect( 'Tool Choice:', options=chat.choice_options,
							key='chat_tool_choice', help=cfg.CHOICE )
						chat.tool_choice = tool_choice
						st.session_state[ 'tool_choice' ] = tool_choice
					
					with llm_four:
						tools = st.multiselect( 'Tools:', options=chat.tool_options, key='chat_tools',
							help=cfg.TOOLS )
						chat.tools = tools
						st.session_state[ 'tools' ] = tools
					
					with llm_five:
						reasoning = st.multiselect( 'Reasoning:', options=chat.reasoning_options,
							key='chat_reasoning', help=cfg.REASONING )
						chat.reasoning = reasoning
					
			# ------------------------------------------------------------------
			# Expander — Hyperparmaters
			# ------------------------------------------------------------------
			with st.expander( 'Inference Options', expanded=False, width='stretch' ):
				prm_one, prm_two, prm_three, prm_four, prm_five = st.columns( [ 0.2, 0.2, 0.2, 0.2, 0.2 ],
					border=True, gap='xsmall'  )
				
				with prm_one:
					top_p = st.slider( 'Top-P', 0.0, 1.0,
						float( st.session_state.get( 'top_p', 1.0 ) ), 0.01, help=cfg.TOP_P )
					st.session_state[ 'top_p' ] = float( top_p )
				
				with prm_two:
					logprobs = st.slider( 'Log-Probs', 0, 20,
						int( st.session_state.get( 'logprobs', 0 ) ), 1, help=cfg.LOG_PROBS )
					st.session_state[ 'logprobs' ] = int( logprobs )
					
				with prm_three:
					freq_penalty = st.slider( 'Frequency Penalty', -2.0, 2.0,
						float( st.session_state.get( 'freq_penalty', 0.0 ) ),
						0.01, help=cfg.FREQUENCY_PENALTY )
					st.session_state[ 'freq_penalty' ] = float( freq_penalty )
				
				with prm_four:
					pres_penalty = st.slider( 'Presence Penalty', -2.0, 2.0,
						float( st.session_state.get( 'pres_penalty', 0.0 ) ),
						0.01, help=cfg.PRESENCE_PENALTY )
					st.session_state[ 'pres_penalty' ] = float( pres_penalty )
				
				with prm_five:
					temperature = st.slider( 'Temperature', 0.0, 1.0,
						float( st.session_state.get( 'temperature', 0.7 ) ), 0.01,
						help=cfg.TEMPERATURE )
					st.session_state[ 'temperature' ] = float( temperature )
			
			# ------------------------------------------------------------------
			# Expander — Response
			# ------------------------------------------------------------------
			with st.expander( 'Response Options', expanded=False, width='stretch' ):
					res_one, res_two, res_three, res_four, res_five = st.columns( [0.20, 0.20, 0.20, 0.20, 0.20 ],
						border=True, gap='small' )
					
					with res_one:
						stream = st.toggle( 'Stream', key='chat_stream', value=False, help=cfg.STREAM )
						st.session_state[ 'stream' ] = stream
						
					with res_two:
						store = st.toggle( 'Store', key='chat_store', value=True, help=cfg.STORE )
						st.session_state[ 'store' ] = store
						
					with res_three:
						back = st.toggle( 'Background', key='chat_bakground',
							value=False, help=cfg.BACKGROUND_MODE )
						st.session_state[ 'background' ] = back
						
					with res_four:
						stop_text = st.text_input( 'Stop Sequences',
							value='\n'.join( st.session_state.get( 'stop_sequences', [ ] ) ),
							help=cfg.STOP_SEQUENCE, width='stretch' )
						
						st.session_state[ 'stop_sequences' ] = [
								s for s in stop_text.splitlines( ) if s.strip( ) ]
					
					with res_five:
						max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=100000,
							value=6048, help=cfg.MAX_OUTPUT_TOKENS )
						st.session_state[ 'max_tokens' ] = int( max_tokens )
			
		# ------------------------------------------------------------------
		# Expander — Instructions
		# ------------------------------------------------------------------
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			st.text_area( 'Prompt Text', height=240, width='stretch',
				help=cfg.SYSTEM_INSTRUCTIONS, key='text_system_instructions' )
			
			if st.button( 'Clear Instructions', width='stretch' ):
				st.session_state[ 'do_clear_instructions' ] = True
				st.rerun( )
		
		# ------------------------------------------------------------------
		# ----------- MESSAGES -----------------------
		# ------------------------------------------------------------------
		for msg in st.session_state.messages:
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
			st.session_state.messages.append( { 'role': 'user', 'content': prompt } )
			with st.chat_message( 'assistant', avatar="" ):
				gen_kwargs = { }
				
				with st.spinner( 'Thinking…' ):
					gen_kwargs[ 'model' ] = st.session_state[ 'text_model' ]
					gen_kwargs[ 'top_p' ] = st.session_state[ 'top_p' ]
					gen_kwargs[ 'logprobs' ] = st.session_state[ 'logprobs' ]
					gen_kwargs[ 'max_tokens' ] = st.session_state[ 'max_tokens' ]
					gen_kwargs[ 'frequency' ] = st.session_state[ 'freq_penalty' ]
					gen_kwargs[ 'presence' ] = st.session_state[ 'pres_penalty' ]
					
					if st.session_state[ 'stop_sequences' ]:
						gen_kwargs[ 'stops' ] = st.session_state[ 'stop_sequences' ]
					
					response = None
					
					try:
						mdl = str( gen_kwargs[ 'model' ] )
						if mdl.startswith( 'gpt-5' ):
							response = chat.generate_text(
								prompt=prompt,
								model=gen_kwargs[ 'model' ]
							)
						else:
							response = chat.generate_text( )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
						response = None
					
					if response is not None and str( response ).strip( ):
						st.markdown( response )
						st.session_state.messages.append( { 'role': 'assistant', 'content': response } )
					else:
						st.error( 'Generation returned no content.' )
			
						try:
							_update_token_counters( getattr( chat, 'response', None ) or response )
						except Exception:
							pass
			
		# --------------------------------------------------------------
		# Token Usage Reporting
		# --------------------------------------------------------------
		lcu = st.session_state.last_call_usage
		tu = st.session_state.token_usage
		
		if any( lcu.values( ) ):
			st.info(
				f'Last call — prompt: {lcu[ "prompt_tokens" ]}, '
				f'completion: {lcu[ "completion_tokens "]}, '
				f'total: {lcu[ "total_tokens" ]}'
			)
		
		if tu[ 'total_tokens' ] > 0:
			st.write( f'Session totals — prompt: {tu[ "prompt_tokens" ]} · '
				f'completion: {tu[ "completion_tokens" ]} · '
				f'total: {tu[ "total_tokens" ]}' )

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == "Images":
	st.subheader( '📷 Images API')
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	image_size = st.session_state.get( 'image_size', None )
	image_style = st.session_state.get( 'image_style', None )
	image_quality = st.session_state.get( 'image_quality', None )
	image_format = st.session_state.get( 'image_format', None )
	image_detail = st.session_state.get( 'image_detail', None )
	image = provider_module.Images( )
	
	# ------------------------------------------------------------------
	# Sidebar — Image Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.text( '⚙️ Image Settings' )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	# ------------------------------------------------------------------
	# EXPANDER — IMAGE SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_image_instructions', False ):
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'clear_image_instructions' ] = False
	
	with center:
		with st.expander( '🧠 LLM Configuration', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Expander — LLM
			# ------------------------------------------------------------------
			with st.expander( 'Model Options', expanded=False, width='stretch' ):
				img_one, img_two, img_three, img_four, img_five = st.columns( [ 0.2, 0.2, 0.2, 0.2, .2 ], border=True )
				
				with img_one:
					# ---------------- Model ---------------
					image_model = st.selectbox( "Image Model", image.model_options,
						index=(image.model_options.index( st.session_state[ "image_model" ] )
						       if st.session_state.get( "image_model" ) in image.model_options
						       else 0), )
					
					st.session_state[ "image_model" ] = image_model
				
				with img_two:
					# ---------------- Size / Aspect Ratio (provider-aware) ----------------
					if hasattr( image, "aspect_options" ):
						size_or_aspect = st.selectbox( "Aspect Ratio", image.aspect_options, )
						size_arg = size_or_aspect
					else:
						size_or_aspect = st.selectbox( "Size", image.size_options, )
						size_arg = size_or_aspect
				
				with img_three:
					# ---------------- Quality ----------------
					quality = None
					if hasattr( image, 'quality_options' ):
						quality = st.selectbox( 'Image Quality', image.quality_options, )
				
				with img_four:
					# ---------------- Format ----------------
					fmt = None
					if hasattr( image, 'format_options' ):
						fmt = st.selectbox( 'Image Format', image.format_options, )
				
				with img_five:
					# ---------------- Detail ----------------
					fmt = None
					if hasattr( image, 'detail_options' ):
						fmt = st.selectbox( 'Image Detail', image.detail_options, )
			
			# ------------------------------------------------------------------
			# Expander — Hyperparmaters
			# ------------------------------------------------------------------
			with st.expander( 'Inference Options', expanded=False, width='stretch' ):
				prm_one, prm_two, prm_three, prm_four, prm_five = st.columns( [ 0.2,
				                                                                0.2,
				                                                                0.2,
				                                                                0.2,
				                                                                0.2 ],
					border=True, gap='xsmall' )
				
				with prm_one:
					top_p = st.slider( 'Top-P', 0.0, 1.0,
						float( st.session_state.get( 'top_p', 1.0 ) ), 0.01, help=cfg.TOP_P )
					st.session_state[ 'top_p' ] = float( top_p )
				
				with prm_two:
					logprobs = st.slider( 'Log-Probs', 0, 20,
						int( st.session_state.get( 'logprobs', 0 ) ), 1, help=cfg.LOG_PROBS )
					st.session_state[ 'logprobs' ] = int( logprobs )
				
				with prm_three:
					freq_penalty = st.slider( 'Frequency Penalty', -2.0, 2.0,
						float( st.session_state.get( 'freq_penalty', 0.0 ) ),
						0.01, help=cfg.FREQUENCY_PENALTY )
					st.session_state[ 'freq_penalty' ] = float( freq_penalty )
				
				with prm_four:
					pres_penalty = st.slider( 'Presence Penalty', -2.0, 2.0,
						float( st.session_state.get( 'pres_penalty', 0.0 ) ),
						0.01, help=cfg.PRESENCE_PENALTY )
					st.session_state[ 'pres_penalty' ] = float( pres_penalty )
				
				with prm_five:
					temperature = st.slider( 'Temperature', 0.0, 1.0,
						float( st.session_state.get( 'temperature', 0.7 ) ), 0.01,
						help=cfg.TEMPERATURE )
					st.session_state[ 'temperature' ] = float( temperature )
			
			# ------------------------------------------------------------------
			# Expander — Response
			# ------------------------------------------------------------------
			with st.expander( 'Response Options', expanded=False, width='stretch' ):
				res_one, res_two, res_three, res_four, res_five = st.columns( [ 0.20,
				                                                                0.20,
				                                                                0.20,
				                                                                0.20,
				                                                                0.20 ],
					border=True, gap='small' )
				
				with res_one:
					stream = st.toggle( 'Stream', key='image_stream', value=False, help=cfg.STREAM )
					st.session_state[ 'stream' ] = stream
				
				with res_two:
					store = st.toggle( 'Store', key='image_store', value=True, help=cfg.STORE )
					st.session_state[ 'store' ] = store
				
				with res_three:
					back = st.toggle( 'Background', key='image_bakground',
						value=False, help=cfg.BACKGROUND_MODE )
					st.session_state[ 'background' ] = back
				
				with res_four:
					n = st.number_input( 'Number', min_value=1, max_value=100,
						value=1, help='Number of generated images' )
					st.session_state[ 'number' ] = int( n )
				
				with res_five:
					max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=100000,
						value=6048, help=cfg.MAX_OUTPUT_TOKENS )
					st.session_state[ 'max_tokens' ] = int( max_tokens )
			
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			st.text_area( 'Prompt Text', height=100, width='stretch',
				help=cfg.SYSTEM_INSTRUCTIONS, key='image_system_instructions' )
			
			if st.button( 'Clear Instructions', width='stretch' ):
				st.session_state[ 'clear_image_instructions' ] = True
				st.rerun( )
	
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
				for candidate in (
							'edit',
							'describe_image',
							'describe',
							'classify',
							'detect_objects',
							'caption',
							'image_edit',
				):
					if hasattr( image, candidate ):
						available_methods.append( candidate )
				
				if available_methods:
					chosen_method = st.selectbox( 'Method', available_methods, index=0, )
				else:
					chosen_method = None
					st.info( 'No dedicated image editing method found on Image object;'
					         'attempting generic handlers.')
				
				chosen_model = st.selectbox( "Model (edit)", [ image_model,  None ], index=0, )
				
				chosen_model_arg = ( image_model if chosen_model is None else chosen_model )
				
				if st.button( "Edit Image" ):
					with st.spinner( "Editing image…" ):
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
								st.warning( "No editing output returned by the available methods." )
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
	provider_name = st.session_state.get( 'provider', 'GPT' )
	audio_file = st.session_state.get( 'audio_file', None )
	audio_rate = st.session_state.get( 'audio_rate', 16000 )
	audio_start = st.session_state.get( 'audio_start', 0.0 )
	audio_end = st.session_state.get( 'audio_end', 0.0 )
	audio_loop = st.session_state.get( 'audio_loop', False )
	auto_play = st.session_state.get( 'auto_play', False )
	voice = st.session_state.get( 'voice', None )
	st.subheader( '🔉 Audio API')
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
	language = None
	voice = None
	audio_model = None
	model_options = [ ]
	
	if transcriber is not None:
		available_tasks.append( 'Transcribe' )
	if translator is not None:
		available_tasks.append( 'Translate' )
	if tts is not None:
		available_tasks.append( 'Text-to-Speech' )
		
	# ------------------------------------------------------------------
	#  AUDIO SETTINGS
	# ------------------------------------------------------------------
	if st.session_state.get( 'clear_audio_instructions', False ):
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'clear_audio_instructions' ] = False
		
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9,  0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — LLM
		# ------------------------------------------------------------------
		with st.expander( '🧠 LLM Configuration', expanded=False, width='stretch' ):
			with st.expander( 'Model Options', expanded=False, width='stretch' ):
				aud_one, aud_two, aud_three, aud_four, aud_five = \
					st.columns( [ 0.2, 0.2, 0.2, 0.2, .2 ], border=True )
				
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
					audio_format = st.selectbox( label='Format',
						options=[ 'audio/mp3',
						          'audio/wav',
						          'audio/aac',
						          'audio/flac',
						          'audio/opus',
						          'audio/pcm' ] )
			
			# ------------------------------------------------------------------
			# Expander — Hyperparmaters
			# ------------------------------------------------------------------
			with st.expander( 'Inference Options', expanded=False, width='stretch' ):
				prm_one, prm_two, prm_three, prm_four, prm_five = st.columns( [ 0.2,
				                                                                0.2,
				                                                                0.2,
				                                                                0.2,
				                                                                0.2 ],
					border=True, gap='xsmall' )
				
				with prm_one:
					top_p = st.slider( 'Top-P', 0.0, 1.0,
						float( st.session_state.get( 'top_p', 1.0 ) ), 0.01, help=cfg.TOP_P )
					st.session_state[ 'top_p' ] = float( top_p )
				
				with prm_two:
					logprobs = st.slider( 'Log-Probs', 0, 20,
						int( st.session_state.get( 'logprobs', 0 ) ), 1, help=cfg.LOG_PROBS )
					st.session_state[ 'logprobs' ] = int( logprobs )
				
				with prm_three:
					freq_penalty = st.slider( 'Frequency Penalty', -2.0, 2.0,
						float( st.session_state.get( 'freq_penalty', 0.0 ) ),
						0.01, help=cfg.FREQUENCY_PENALTY )
					st.session_state[ 'freq_penalty' ] = float( freq_penalty )
				
				with prm_four:
					pres_penalty = st.slider( 'Presence Penalty', -2.0, 2.0,
						float( st.session_state.get( 'pres_penalty', 0.0 ) ),
						0.01, help=cfg.PRESENCE_PENALTY )
					st.session_state[ 'pres_penalty' ] = float( pres_penalty )
				
				with prm_five:
					temperature = st.slider( 'Temperature', 0.0, 1.0,
						float( st.session_state.get( 'temperature', 0.7 ) ), 0.01,
						help=cfg.TEMPERATURE )
					st.session_state[ 'temperature' ] = float( temperature )
			
			# ------------------------------------------------------------------
			# Expander — Response
			# ------------------------------------------------------------------
			with st.expander( 'Response Options', expanded=False, width='stretch' ):
				resp_one, resp_two, resp_three, resp_four, resp_five = \
					st.columns( [ 0.20, 0.20, 0.20, 0.20, 0.20 ],
					border=True, )
				
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
		
		with st.expander( '🖥️ System Instructions', expanded=False, width='stretch' ):
			st.text_area( 'Prompt Text', height=100, width='stretch',
				help=cfg.SYSTEM_INSTRUCTIONS, key='audio_system_instructions' )
			
			if st.button( 'Clear Instructions', width='stretch' ):
				st.session_state[ 'clear_audio_instructions' ] = True
				st.rerun( )

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
			if audio_file is not None:
				array = np.ndarray( audio_file )
			else:
				array = None
			audio_recording = st.audio( data=audio_file, sample_rate=audio_rate,
				start_time=audio_start, end_time=audio_end, format=audio_format, width='stretch',
				loop=audio_loop, autoplay=auto_play  )
	
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
						else 0 ), )
			
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
							vector = embed.create( text, model=embed_model, method=method, )
						else:
							vector = embed.create( text, model=embed_model, )
						
						st.write( 'Vector length:', len( vector ) )
						
						try:
							_update_token_counters( getattr( embed, 'response', None ) )
						except Exception:
							pass
					
					except Exception as exc:
						st.error( f'Embedding failed: {exc}' )

# ======================================================================================
# VECTOR MODE
# ======================================================================================
elif mode == 'Vector Stores':
	provider_name = st.session_state.get( 'provider', 'GPT' )
	# ------------------------------------------------------------------
	# Provider-aware VectorStore instantiation
	# ------------------------------------------------------------------
	vector = None
	collector  = None
	searcher = None
	
	if provider_name == 'Grok':
		provider_module = get_provider_module( )
		collector = provider_module.VectorStores( )
	
		st.subheader( '📚 Collections' )
		st.divider( )
		
		# --------------------------------------------------------------
		# Local mapping (if maintained by wrapper)
		# --------------------------------------------------------------
		vs_map = getattr( collector, "collections", None )
		if vs_map and isinstance( vs_map, dict ):
			st.markdown( "**Known Collections (local mapping)**" )
			for name, vid in vs_map.items( ):
				st.write( f"- **{name}** — `{vid}`" )
			st.markdown( "---" )
		
		# --------------------------------------------------------------
		# Create Collection
		# --------------------------------------------------------------
		with st.expander( "Create Collection", expanded=False ):
			new_store_name = st.text_input( "New Collection Name" )
			if st.button( "➕ Create Collection" ):
				if not new_store_name:
					st.warning( "Enter a Collection Name." )
				else:
					try:
						if hasattr( collector, "create" ):
							res = provider_module.create( new_store_name )
							st.success(
								f"Create call submitted for '{new_store_name}'."
							)
						else:
							st.warning( "create() not available on Grok provider." )
					except Exception as exc:
						st.error( f"Create collection failed: {exc}" )
		
		# --------------------------------------------------------------
		# Discover collections (local → API fallback)
		# --------------------------------------------------------------
		options: List[ tuple ] = [ ]
		if vs_map and isinstance( vs_map, dict ):
			options = list( vs_map.items( ) )
		
		if not options:
			try:
				client = getattr( collector, "client", None )
				if (
						client
						and hasattr( client, "collections" )
						and hasattr( client.collections, "list" )
				):
					api_list = client.collections.list( )
					temp: List[ tuple ] = [ ]
					for item in getattr( api_list, "data", [ ] ) or api_list:
						nm = getattr( item, "name", None ) or (
								item.get( "name" ) if isinstance( item, dict ) else None
						)
						vid = getattr( item, "id", None ) or (
								item.get( "id" ) if isinstance( item, dict ) else None
						)
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
			sel = st.selectbox( "Select a Collection", options=names )
			
			sel_id: Optional[ str ] = None
			for n, i in options:
				if f"{n} — {i}" == sel:
					sel_id = i
					break
			
			c1, c2 = st.columns( [ 1,  1 ] )
			
			with c1:
				if st.button( "Retrieve Collection" ):
					if not sel_id:
						st.warning( "No collection selected." )
					else:
						try:
							client = getattr( collector, "client", None )
							if (
									client
									and hasattr( client, "collections" )
									and hasattr( client.collections, "retrieve" )
							):
								vs = client.collections.retrieve( collection_id=sel_id )
								st.json(
									vs.__dict__
									if hasattr( vs, "__dict__" )
									else vs
								)
							else:
								st.warning( "collections.retrieve() not available." )
						except Exception as exc:
							st.error( f"retrieve() failed: {exc}" )
			
			with c2:
				if st.button( "❌ Delete Collection" ):
					if not sel_id:
						st.warning( "No collection selected." )
					else:
						try:
							client = getattr( collector, "client", None )
							if (
									client
									and hasattr( client, "collections" )
									and hasattr( client.collections, "delete" )
							):
								res = client.collections.delete(
									collection_id=sel_id
								)
								st.success( f"Delete returned: {res}" )
							else:
								st.warning(
									"collections.delete() not available."
								)
						except Exception as exc:
							st.error( f"Delete failed: {exc}" )
				else:
					st.info(
						"No collections discovered. Create one or confirm "
						"collections exist for this account." )
						
	elif provider_name == 'Gemini':
		provider_module = get_provider_module( )
		searcher = provider_module.VectorStores( )
	
		st.subheader( '🔍 File Search Stores' )
		st.divider( )
		
		# --------------------------------------------------------------
		# Local mapping (if maintained by wrapper)
		# --------------------------------------------------------------
		vs_map = getattr( searcher, 'collections', None )
		if vs_map and isinstance( vs_map, dict ):
			st.markdown( 'Known File Search Stores (local mapping)' )
			for name, vid in vs_map.items( ):
				st.write( f'- **{name}** — {vid}' )
			st.divider( )
		
		# --------------------------------------------------------------
		# Create File Search Store
		# --------------------------------------------------------------
		with st.expander( 'Create File Search Store', expanded=False ):
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
				if (
						client
						and hasattr( client, 'file_search_stores' )
						and hasattr( client.file_search_stores, 'list' )
				):
					api_list = client.file_search_stores.list( )
					temp: List[ tuple ] = [ ]
					for item in getattr( api_list, 'data', [ ] ) or api_list:
						nm = getattr( item, 'name', None ) or (
								item.get( 'name' ) if isinstance( item, dict ) else None
						)
						vid = getattr( item, 'id', None ) or (
								item.get( 'id' ) if isinstance( item, dict ) else None
						)
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
				if st.button( 'Retrieve File Search Store' ):
					if not sel_id:
						st.warning( 'No file search store selected.' )
					else:
						try:
							client = getattr( searcher, 'client', None )
							if (
									client
									and hasattr( client, 'file_search_stores' )
									and hasattr( client.file_search_stores, 'retrieve' )
							):
								vs = client.file_search_stores.retrieve(
									file_search_store_id=sel_id )
								st.json( vs.__dict__ if hasattr( vs, '__dict__' ) else vs )
							else:
								st.warning( 'file_search_stores.retrieve() not available.' )
						except Exception as exc:
							st.error( f'retrieve() failed: {exc}' )
			
			with c2:
				if st.button( '❌ Delete File Search Store' ):
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
	
		st.subheader( '⚡ Vector Stores' )
		st.divider( )
		
		# --------------------------------------------------------------
		# Local mapping
		# --------------------------------------------------------------
		vs_map = getattr( vector, 'collections', None )
		if vs_map and isinstance( vs_map, dict ):
			st.markdown( '**Known Vector Stores (local mapping)**' )
			for name, vid in vs_map.items( ):
				st.write( f'- **{name}** — `{vid}`' )
			st.divider( )
		
		# --------------------------------------------------------------
		# Create Vector Store
		# --------------------------------------------------------------
		with st.expander( 'Create Vector Store', expanded=False ):
			new_store_name = st.text_input( 'New Vector Store name' )
			if st.button( '➕ Create Vector Store' ):
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
		
		# --------------------------------------------------------------
		# Discover vector stores
		# --------------------------------------------------------------
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
								item.get( 'name' ) if isinstance( item, dict ) else None
						)
						vid = getattr( item, 'id', None ) or (
								item.get( 'id' ) if isinstance( item, dict ) else None
						)
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
			sel = st.selectbox( 'Select a vector store', options=names )
			
			sel_id: Optional[ str ] = None
			for n, i in options:
				if f'{n} — {i}' == sel:
					sel_id = i
					break
			
			c1, c2 = st.columns( [ 1, 1 ] )
			
			with c1:
				if st.button( 'Retrieve Vector Store' ):
					if not sel_id:
						st.warning( 'No vector store selected.' )
					else:
						try:
							openai_client = st.session_state[ 'openai_client' ]
							
							vs = openai_client.vector_stores.retrieve(
								vector_store_id=sel_id
							)
							
							st.json( vs.model_dump( ) )
							st.write( 'Name:', data[ 'name' ] )
							st.write( 'Files:', data[ 'file_counts' ][ 'completed' ] )
							st.write( 'Size (MB):', round( data[ 'usage_bytes' ] / 1_048_576, 2 ) )
						except Exception as exc:
							st.error( f'retrieve() failed: {exc}' )
			
			with c2:
				if st.button( '❌ Delete Vector Store' ):
					if not sel_id:
						st.warning( 'No vector store selected.' )
					else:
						try:
							openai_client = st.session_state.get( 'openai_client' )
							if openai_client and hasattr( openai_client.vector_stores, 'delete' ):
								res = openai_client.vector_stores.delete(
									vector_store_id=sel_id
								)
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
	st.subheader( '📚 Document Q & A' )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — Inference Parameters
		# ------------------------------------------------------------------
		with st.expander( '🧠 Inference Options', expanded=False, width='stretch' ):
			inf_one, inf_two, inf_three, inf_four, inf_five = \
				st.columns( [ 0.2, 0.2, 0.2,  0.2, 0.2 ],
				border=True, gap='xsmall' )
			
			with inf_one:
				top_p = st.slider( 'Top-P', 0.0, 1.0,
					float( st.session_state.get( 'top_p', 1.0 ) ), 0.01, help=cfg.TOP_P )
				st.session_state[ 'top_p' ] = float( top_p )
			
			with inf_two:
				logprobs = st.slider( 'Log-Probs', 0, 20,
					int( st.session_state.get( 'logprobs', 0 ) ), 1, help=cfg.LOG_PROBS )
				st.session_state[ 'logprobs' ] = int( logprobs )
			
			with inf_three:
				freq_penalty = st.slider( 'Frequency Penalty', -2.0, 2.0,
					float( st.session_state.get( 'freq_penalty', 0.0 ) ),
					0.01, help=cfg.FREQUENCY_PENALTY )
				st.session_state[ 'freq_penalty' ] = float( freq_penalty )
			
			with inf_four:
				pres_penalty = st.slider( 'Presence Penalty', -2.0, 2.0,
					float( st.session_state.get( 'pres_penalty', 0.0 ) ),
					0.01, help=cfg.PRESENCE_PENALTY )
				st.session_state[ 'pres_penalty' ] = float( pres_penalty )
			
			with inf_five:
				temperature = st.slider( 'Temperature', 0.0, 1.0,
					float( st.session_state.get( 'temperature', 0.7 ) ), 0.01,
					help=cfg.TEMPERATURE )
				st.session_state[ 'temperature' ] = float( temperature )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( '💻 System Instructions', expanded=False, width='stretch' ):
			left_inst, right_inst = st.columns( [ 0.6, 0.4 ], vertical_alignment='center', border=True )
			
			with left_inst:
				st.text_area( 'Enter Text', height=100, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='doc_system_instruction' )
				instructions = st.session_state.get( 'doc_system_instruction', '' )
				
			with right_inst:
				source = st.radio( 'Source', [ 'Upload Local', 'Files API', 'Vector Store' ] )
				st.session_state.doc_source = source.lower( ).replace( ' ', '' )
			
			left_btn, center_btn, right_btn = st.columns( [ 0.6,  0.2, 0.2 ] )
			with left_btn:
				if st.button( 'Clear Conversation', width='stretch' ):
					st.session_state.doc_messages = [ ]
					st.rerun( )
			
			with center_btn:
				reset_doc_ins = st.button( 'Clear Instructions', width='stretch', key='clear_doc_instructions' )
				if reset_doc_ins:
					st.session_state.doc_system_instruction = ''
					
			with right_btn:
				if st.button( 'Summarize Document', width='stretch' ):
					if not st.session_state.get( 'doc_active_docs' ):
						st.warning( 'No document loaded.' )
					else:
						st.session_state.doc_messages.append({'role': 'user',
						                                      'content': 'Summarize this document.'})
					
					summary = summarize_active_document( )
					st.session_state.doc_messages.append({'role': 'assistant','content': summary})
					
					st.rerun( )
			
		doc_left, doc_right = st.columns( [ 0.2, 0.8 ], border=True )
		with doc_left:
			uploaded = st.file_uploader( 'Upload', type=[ 'pdf', 'txt', 'md', 'docx' ],
				accept_multiple_files=False, label_visibility='visible' )
			
			if uploaded is not None:
				st.session_state.doc_active_docs = [ uploaded.name ]
				st.session_state.doc_bytes = { uploaded.name: uploaded.getvalue( ) }
				st.success( f'{uploaded.name} has been loaded!' )
			else:
				st.info( 'Load a document.' )
			
			unload = st.button( label='Unload Document', width='stretch' )
			if unload:
				uploaded = None
				st.session_state.doc_active_docs = None
		
		with doc_right:
			if st.session_state.get( 'doc_active_docs' ):
				name = st.session_state.doc_active_docs[ 0 ]
				file_bytes = st.session_state.doc_bytes.get( name )
				if file_bytes:
					st.pdf( file_bytes, height=420 )
		
		for msg in st.session_state.doc_messages:
			with st.chat_message( msg[ 'role' ] ):
				st.markdown( msg[ 'content' ] )
		
		if prompt := st.chat_input( 'Ask a question about the document' ):
			st.session_state.doc_messages.append( { 'role': 'user', 'content': prompt } )
			response = route_document_query( prompt )
			st.session_state.doc_messages.append( { 'role': 'assistant', 'content': response } )
			st.rerun( )

# ======================================================================================
# FILES API MODE
# ======================================================================================
elif mode == "Files":
	try:
		chat  # type: ignore
	except NameError:
		chat = get_provider_module( ).Chat( )
	
	st.subheader( '📁 Files API' )
	st.divider( )
	left, center, right = st.columns( [ 0.25,  3.5, 0.25 ] )
	with center:
		list_method = None
		for name in ( 'retrieve_files', 'list_files', 'get_files', ):
			if hasattr( chat, name ):
				list_method = getattr( chat, name )
				break
		
		uploaded_file = st.file_uploader(
			'Upload file (server-side via Files API)',
			type=[ 'pdf', 'txt', 'md', 'docx', 'png', 'jpg', 'jpeg', ],
		)
		if uploaded_file:
			tmp_path = save_temp( uploaded_file )
			upload_fn = None
			for name in ("upload_file", "upload", "files_upload"):
				if hasattr( chat, name ):
					upload_fn = getattr( chat, name )
					break
			if not upload_fn:
				st.warning( "No upload function found on chat object (upload_file)." )
			else:
				with st.spinner( "Uploading to Files API..." ):
					try:
						fid = upload_fn( tmp_path )
						st.success( f"Uploaded; file id: {fid}" )
					except Exception as exc:
						st.error( f"Upload failed: {exc}" )
	
		if st.button( "List files" ):
			try:
				files_resp = list_method( )
				rows = [ ]
				
				files_list = (
						files_resp.data
						if hasattr( files_resp, "data" )
						else files_resp
						if isinstance( files_resp, list )
						else [ ]
				)
				
				for f in files_list:
					rows.append( {
							"id": str( getattr( f, "id", "" ) ),
							"filename": str( getattr( f, "filename", "" ) ),
							"purpose": str( getattr( f, "purpose", "" ) ),
					} )
				
				st.session_state.files_table = rows
			
			except Exception as exc:
				st.session_state.files_table = None
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
	
	TABLE = "Prompts"
	PAGE_SIZE = 10
	
	st.subheader( "📝 Prompt Engineering" )
	st.divider( )
	
	st.session_state.setdefault( "pe_cascade_enabled", False )
	st.checkbox( "Cascade selection into System Instructions", key="pe_cascade_enabled" )
	
	# ------------------------------------------------------------------
	# Session state
	# ------------------------------------------------------------------
	st.session_state.setdefault( "pe_page", 1 )
	st.session_state.setdefault( "pe_search", "" )
	st.session_state.setdefault( "pe_sort_col", "PromptsId" )
	st.session_state.setdefault( "pe_sort_dir", "ASC" )
	st.session_state.setdefault( "pe_selected_id", None )
	
	st.session_state.setdefault( "pe_name", "" )
	st.session_state.setdefault( "pe_text", "" )
	st.session_state.setdefault( "pe_version", 1 )
	
	# ------------------------------------------------------------------
	# DB helpers
	# ------------------------------------------------------------------
	def get_conn( ):
		return sqlite3.connect( cfg.DB_PATH )
	
	def reset_selection( ):
		st.session_state.pe_selected_id = None
		st.session_state.pe_name = ''
		st.session_state.pe_text =''
		st.session_state.pe_version = 1
	
	def load_prompt( pid: int ) -> None:
		with get_conn( ) as conn:
			cur = conn.execute(
				f"SELECT Name, Text, Version FROM {TABLE} WHERE PromptsId=?",
				(pid,),
			)
			row = cur.fetchone( )
			if not row:
				return
			st.session_state.pe_name = row[ 0 ]
			st.session_state.pe_text = row[ 1 ]
			st.session_state.pe_version = row[ 2 ]
	
	# ------------------------------------------------------------------
	# Filters
	# ------------------------------------------------------------------
	c1, c2, c3, c4 = st.columns( [ 4,
	                               2,
	                               2,
	                               3 ] )
	
	with c1:
		st.text_input( "Search (Name/Text contains)", key="pe_search" )
	
	with c2:
		st.selectbox(
			"Sort by",
			[ "PromptsId",
			  "Name",
			  "Version" ],
			key="pe_sort_col",
		)
	
	with c3:
		st.selectbox( "Direction", [ "ASC",
		                             "DESC" ], key="pe_sort_dir" )
	
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
			st.button( "Clear", on_click=reset_selection )
	
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
					"Selected": r[ 0 ] == st.session_state.pe_selected_id,
					"PromptsId": r[ 0 ],
					"Name": r[ 1 ],
					"Version": r[ 3 ],
					"ID": r[ 4 ],
			}
		)
	
	edited = st.data_editor(
		table_rows,
		hide_index=True,
		use_container_width=True,
		key="prompt_table",
	)
	
	# ------------------------------------------------------------------
	# SELECTION PROCESSING (must run BEFORE widgets below)
	# ------------------------------------------------------------------
	selected = [
			r for r in edited
			if isinstance( r, dict ) and r.get( "Selected" )
	]
	
	if len( selected ) == 1:
		pid = int( selected[ 0 ][ "PromptsId" ] )
		if pid != st.session_state.pe_selected_id:
			st.session_state.pe_selected_id = pid
			load_prompt( pid )
	
	elif len( selected ) == 0:
		reset_selection( )
	
	elif len( selected ) > 1:
		st.warning( "Select exactly one prompt row." )
	
	# ------------------------------------------------------------------
	# Paging
	# ------------------------------------------------------------------
	p1, p2, p3 = st.columns( [ 0.25,
	                           3.5,
	                           0.25 ] )
	
	with p1:
		if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
			st.session_state.pe_page -= 1
	
	with p2:
		st.markdown(
			f"Page **{st.session_state.pe_page}** of **{total_pages}**"
		)
	
	with p3:
		if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
			st.session_state.pe_page += 1
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	# ------------------------------------------------------------------
	# Edit Prompt
	# ------------------------------------------------------------------
	with st.expander( "🖊️ Edit Prompt", expanded=True ):
		st.text_input(
			"PromptsId",
			value=st.session_state.pe_selected_id or "",
			disabled=True,
		)
		
		st.text_input( "Name", key="pe_name" )
		st.text_area( "Text", key="pe_text", height=260 )
		st.number_input( "Version", min_value=1, key="pe_version" )
		
		c1, c2, c3 = st.columns( 3 )
		
		with c1:
			if st.button(
					"💾 Save Changes"
					if st.session_state.pe_selected_id
					else "➕ Create Prompt"
			):
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
				
				st.success( "Saved." )
				reset_selection( )
		
		with c2:
			if st.session_state.pe_selected_id and st.button( "Delete" ):
				with get_conn( ) as conn:
					conn.execute(
						f"DELETE FROM {TABLE} WHERE PromptsId=?",
						(st.session_state.pe_selected_id,),
					)
					conn.commit( )
				reset_selection( )
				st.success( "Deleted." )
		
		with c3:
			st.button( "🧹 Clear Selection", on_click=reset_selection )

# ==============================================================================
# EXPORT MODE
# ==============================================================================
elif mode == 'Data Export':
	st.subheader( '📭  Export' )
	st.divider( )
	
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
	st.subheader( "🗄 Data Management" )
	tabs = st.tabs( [
			"📥 Import",
			"🗂 Browse",
			"✏ CRUD",
			"📊 Explore",
			"🔎 Filter",
			"🧮 Aggregate",
			"📈 Visualize",
			"⚙ Admin",
			"🧠 SQL"
	] )
	
	tables = dm_tables( )
	if not tables:
		st.info( "No tables available." )
	else:
		table = st.selectbox( "Table", tables )
		df_full = dm_read( table )
	
	# ------------------------------------------------------------------------------
	# UPLOAD TAB
	# ------------------------------------------------------------------------------
	with tabs[ 0 ]:
		uploaded_file = st.file_uploader( "Upload Excel File", type=[ "xlsx" ] )
		overwrite = st.checkbox( "Overwrite existing tables", value=True )
		if uploaded_file:
			sheets = pd.read_excel( uploaded_file, sheet_name=None )
			for sheet_name, df in sheets.items( ):
				table_name = sheet_name.replace( " ", "_" )
				if overwrite:
					dm_drop_table( table_name )
				
				dm_create_table_from_df( table_name, df )
				dm_insert_df( table_name, df )
			st.success( "Import completed." )
			st.rerun( )
		
	# ------------------------------------------------------------------------------
	# BROWSE TAB
	# ------------------------------------------------------------------------------
	with tabs[ 1 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key='table_name' )
			df = dm_read( table )
			st.dataframe( df, use_container_width=True )
		else:
			st.info( "No tables available." )
		
	# ------------------------------------------------------------------------------
	# CRUD TAB
	# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# CRUD (Schema-Aware)
# ------------------------------------------------------------------------------

with tabs[ 2 ]:
	tables = dm_tables( )
	
	if not tables:
		st.info( "No tables available." )
	else:
		table = st.selectbox( "Select Table", tables, key="crud_table" )
		
		df = dm_read( table )
		schema = dm_schema( table )
		
		# Build type map
		type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != "rowid" }
		
		# ------------------------------------------------------------------
		# INSERT
		# ------------------------------------------------------------------
		st.subheader( "Insert Row" )
		
		insert_data = { }
		
		for column, col_type in type_map.items( ):
			if "INT" in col_type:
				insert_data[ column ] = st.number_input( column, step=1, key=f"ins_{column}" )
			
			elif "REAL" in col_type:
				insert_data[
					column ] = st.number_input( column, format="%.6f", key=f"ins_{column}" )
			
			elif "BOOL" in col_type:
				insert_data[ column ] = 1 if st.checkbox( column, key=f"ins_{column}" ) else 0
			
			else:
				insert_data[ column ] = st.text_input( column, key=f"ins_{column}" )
		
		if st.button( "Insert Row" ):
			cols = list( insert_data.keys( ) )
			placeholders = ", ".join( [ "?" ] * len( cols ) )
			
			stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
			
			with dm_conn( ) as conn:
				conn.execute( stmt, list( insert_data.values( ) ) )
				conn.commit( )
			
			st.success( "Row inserted." )
			st.rerun( )
		
		# ------------------------------------------------------------------
		# UPDATE
		# ------------------------------------------------------------------
		st.subheader( "Update Row" )
		
		rowid = st.number_input( "Row ID", min_value=1, step=1 )
		
		update_data = { }
		
		for column, col_type in type_map.items( ):
			if "INT" in col_type:
				val = st.number_input( column, step=1, key=f"upd_{column}" )
				update_data[ column ] = val
			
			elif "REAL" in col_type:
				val = st.number_input( column, format="%.6f", key=f"upd_{column}" )
				update_data[ column ] = val
			
			elif "BOOL" in col_type:
				val = 1 if st.checkbox( column, key=f"upd_{column}" ) else 0
				update_data[ column ] = val
			
			else:
				val = st.text_input( column, key=f"upd_{column}" )
				update_data[ column ] = val
		
		if st.button( "Update Row" ):
			set_clause = ", ".join( [ f"{c}=?" for c in update_data ] )
			stmt = f'UPDATE "{table}" SET {set_clause} WHERE rowid=?;'
			
			with dm_conn( ) as conn:
				conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
				conn.commit( )
			
			st.success( "Row updated." )
			st.rerun( )
		
		# ------------------------------------------------------------------
		# DELETE
		# ------------------------------------------------------------------
		st.subheader( "Delete Row" )
		
		delete_id = st.number_input( "Row ID to Delete", min_value=1, step=1 )
		
		if st.button( "Delete Row" ):
			with dm_conn( ) as conn:
				conn.execute( f'DELETE FROM "{table}" WHERE rowid=?;', (delete_id,) )
				conn.commit( )
			
			st.success( "Row deleted." )
			st.rerun( )
	# ------------------------------------------------------------------------------
	# EXPLORE
	# ------------------------------------------------------------------------------
	with tabs[ 3 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key="explore_table" )
			page_size = st.slider( "Rows per page", 10, 500, 50 )
			page = st.number_input( "Page", min_value=1, step=1 )
			offset = (page - 1) * page_size
			df_page = dm_read( table, page_size, offset )
			st.dataframe( df_page, use_container_width=True )
		
	# ------------------------------------------------------------------------------
	# FILTER
	# ------------------------------------------------------------------------------
	with tabs[ 4 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key="filter_table" )
			df = dm_read( table )
			
			column = st.selectbox( "Column", df.columns )
			value = st.text_input( "Contains" )
			
			if value:
				df = df[ df[ column ].astype( str ).str.contains( value ) ]
			st.dataframe( df, use_container_width=True )
		
	# ------------------------------------------------------------------------------
	# AGGREGATE
	# ------------------------------------------------------------------------------
	with tabs[ 5 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key="agg_table" )
			df = dm_read( table )
			numeric_cols = df.select_dtypes( include=[ "number" ] ).columns.tolist( )
			
			if numeric_cols:
				col = st.selectbox( "Column", numeric_cols )
				agg = st.selectbox( "Function", [ "SUM",
				                                  "AVG",
				                                  "COUNT" ] )
				if agg == "SUM":
					st.metric( "Result", df[ col ].sum( ) )
				elif agg == "AVG":
					st.metric( "Result", df[ col ].mean( ) )
				elif agg == "COUNT":
					st.metric( "Result", df[ col ].count( ) )
		
	# ------------------------------------------------------------------------------
	# VISUALIZE
	# ------------------------------------------------------------------------------
	with tabs[ 6 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key="viz_table" )
			df = dm_read( table )
			numeric_cols = df.select_dtypes( include=[ "number" ] ).columns.tolist( )
			
			if numeric_cols:
				col = st.selectbox( "Column", numeric_cols )
				fig = px.histogram( df, x=col )
				st.plotly_chart( fig, use_container_width=True )
		
	# ------------------------------------------------------------------------------
	# ADMIN
	# ------------------------------------------------------------------------------
	with tabs[ 7 ]:
		tables = dm_tables( )
		if tables:
			table = st.selectbox( "Table", tables, key="admin_table" )
			
			if st.button( "Drop Table" ):
				confirm = st.checkbox( "Confirm Drop" )
				if confirm:
					dm_drop_table( table )
					st.success( "Dropped." )
					st.rerun( )
			
			df = dm_read( table )
			col = st.selectbox( "Create Index On", df.columns )
			
			if st.button( "Create Index" ):
				dm_create_index( table, col )
				st.success( "Index created." )
	st.divider( )
	st.subheader( "Create Custom Table" )
	
	new_table_name = st.text_input( "Table Name" )
	
	column_count = st.number_input( "Number of Columns", min_value=1, max_value=20, value=1 )
	
	columns = [ ]
	
	for i in range( column_count ):
		st.markdown( f"### Column {i + 1}" )
		
		col_name = st.text_input( "Column Name", key=f"col_name_{i}" )
		
		col_type = st.selectbox(
			"Column Type",
			[ "INTEGER",
			  "REAL",
			  "TEXT" ],
			key=f"col_type_{i}"
		)
		
		not_null = st.checkbox( "NOT NULL", key=f"not_null_{i}" )
		primary_key = st.checkbox( "PRIMARY KEY", key=f"pk_{i}" )
		auto_inc = st.checkbox( "AUTOINCREMENT (INTEGER only)", key=f"ai_{i}" )
		
		columns.append( {
				"name": col_name,
				"type": col_type,
				"not_null": not_null,
				"primary_key": primary_key,
				"auto_increment": auto_inc
		} )
	
	if st.button( "Create Table" ):
		try:
			dm_create_custom_table( new_table_name, columns )
			st.success( "Table created successfully." )
			st.rerun( )
		
		except Exception as e:
			st.error( f"Error: {e}" )
	# ------------------------------------------------------------------------------
	# SQL
	# ------------------------------------------------------------------------------
	with tabs[ 8 ]:
		query = st.text_area( "SELECT only (single statement)" )
		if st.button( "Run Query" ):
			if ";" in query.strip( )[ :-1 ]:
				st.error( "Multiple statements not allowed." )
			elif not query.lower( ).strip( ).startswith( "select" ):
				st.error( "Only SELECT allowed." )
			else:
				with dm_conn( ) as conn:
					result = pd.read_sql_query( query, conn )
				st.dataframe( result, use_container_width=True )
				
# ======================================================================================
# Footer — Fixed Bottom Status Bar
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
		color: #3e9cfa;
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
	unsafe_allow_html=True, )

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
	None, )

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
		unsafe_allow_html=True, )
