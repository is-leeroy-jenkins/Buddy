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
import base64
import io
import multiprocessing
import os
import sqlite3
from pathlib import Path
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

APP_TITLE = 'Buddy'

APP_SUBTITLE = 'AI for Budget Analysts'

PROMPT_ID = 'pmpt_697f53f7ddc881938d81f9b9d18d6136054cd88c36f94549'

PROMPT_VERSION = '3'

TEXT_TYPES = { 'output_text' }

MARKDOWN_HEADING_PATTERN: re.Pattern[ str ] = re.compile( r"^##\s+(?P<title>.+?)\s*$" )

XML_BLOCK_PATTERN: re.Pattern[ str ] = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>",
	re.DOTALL )

DB_PATH = "stores/sqlite/Data.db"

DEFAULT_CTX = 4096

CPU_CORES = multiprocessing.cpu_count( )

ANALYST = '❓'

BUDDY = r''
# ==============================================================================
# Configuration
# ==============================================================================
client = OpenAI( )

# ==============================================================================
# Page Setup
# ==============================================================================
st.logo( LOGO, size='large', link=CRS )
st.set_page_config( page_title=APP_TITLE, layout="wide",
	page_icon=FAVICON, initial_sidebar_state='collapsed' )
st.caption( APP_SUBTITLE )

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
				current_tag = ( title.strip( ) .lower( )
						.replace( ' ', '_' ) .replace( '-', '_' ) )
			else:
				if current_tag is not None:
					buffer.append( line )	
		flush( )
		
		# Remove trailing whitespace blocks
		while output and not output[ -1 ].strip( ):
			output.pop( )
		return "\n".join( output )                                              

# --------------------------------------------------------------------------------------

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

def image_to_avatar_b64( path: str ) -> str:
	"""
		Purpose:
		--------
		Convert an image file into a Base64 data URI suitable for Streamlit chat avatars.
	
		Parameters:
		-----------
		path (str): Path to the image file.
	
		Returns:
		--------
		str: Base64-encoded data URI.
	"""
	data = Path( path ).read_bytes( )
	encoded = base64.b64encode( data ).decode( "utf-8" )
	return f"data:image/png;base64,{encoded}"

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
		if t not in ('web_search_call', 'file_search_call'):
			continue
		
		raw = None
		if getattr( item, 'sources', None ):
			raw = item.sources
		elif getattr( item, 'action', None ) and getattr( item.action, 'sources', None ):
			raw = item.action.sources
		
		if raw:
			for src in raw:
				s = normalize( src )
				sources.append(
					{
							'title': s.get( 'title' ),
							'snippet': s.get( 'snippet' ),
							'url': s.get( 'url' ),
							'file_id': s.get( 'file_id' ),
					}
				)
	
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
	
	# ---------------------
	# Model initialization
	ctx: int = st.slider( 'Context Window',
		min_value=2048, max_value=8192, value=DEFAULT_CTX, step=512,
		help=( 'Maximum number of tokens the model uses: including system instructions, '
		       'history, and context.' ), )
	
	threads: int = st.slider( 'CPU Threads',
		min_value=1, max_value=CPU_CORES, value=CPU_CORES, step=1,
		help=(
				'Number of CPU threads used for inference; higher values improve speed but increase CPU '
				'usage.' ), )
	
	# ---------------------
	# Inference parameters
	max_tokens: int = st.slider( 'Max Tokens',
		min_value=128, max_value=4096, value=1024, step=128,
		help='Maximum number of tokens generated per response.', )
	
	temperature: float = st.slider( 'Temperature',
		min_value=0.1, max_value=1.5, value=0.7, step=0.1,
		help=(
				'Controls randomness in generation; lower values are more deterministic, higher values '
				'increase creativity.' ), )
	
	top_p: float = st.slider( 'Top-p',
		min_value=0.1, max_value=1.0, value=0.9, step=0.05,
		help=(
				'Nucleus sampling threshold; limits token selection to the smallest set whose cumulative '
				'probability exceeds this value.' ), )
	
	top_k: int = st.slider( 'Top-k',
		min_value=1, max_value=50, value=5, step=1,
		help='Limits token selection to the top K most probable tokens at each step.', )
	
	repeat_penalty: float = st.slider( 'Repeat Penalty',
		min_value=1.0, max_value=2.0, value=1.1, step=0.05,
		help='Penalizes repeated tokens to reduce looping and redundant responses.', )
	
	repeat_last_n: int = st.slider( 'Repeat Window',
		min_value=0, max_value=1024, value=64, step=16,
		help='Number of recent tokens considered for repetition penalties; 0 disables the window.', )
	
	presence_penalty: float = st.slider( 'Presence Penalty',
		min_value=0.0, max_value=2.0, value=0.0, step=0.05,
		help='Encourages introducing new topics by penalizing tokens already present in the context.', )
	
	frequency_penalty: float = st.slider( 'Frequency Penalty',
		min_value=0.0, max_value=2.0, value=0.0, step=0.05,
		help='Reduces repeated phrasing by penalizing tokens based on how often they appear.', )
	
	seed: int = st.number_input( 'Random Seed', value=-1, step=1,
		help='Set to a fixed value for reproducible outputs; use -1 for a random seed each run.', )
	
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
tab_chat, tab_guidance, tab_analysis, tab_system, tab_basic, tab_semantic, tab_prompt, tab_export = st.tabs(
	[ '🧠 Chat', '🏛️ Guidance', '📈 Analysis', '🧬 System Instructions', '📚 Retrieval Augmentation', '🔍 Semantic Search',
	  '📝 Prompt Engineering', '💻 Data Export' ] )

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

# ============================================================================
# CHAT TAB
# ============================================================================
with tab_chat:
	st.subheader( '' )
	for msg in st.session_state.chat_history:
		with st.chat_message( msg[ "role" ] ):
			st.markdown( msg[ "content" ] )
	
	user_input = st.chat_input( "Got a Planning, Programming, or Budget Execution question?..." )
	if user_input:
		st.session_state.chat_history.append(
			{
					'role': 'user',
					'content': user_input }
		)
		
		with st.chat_message( 'user' ):
			st.markdown( user_input )
		
		full_input = build_intent_prefix(
			st.session_state.execution_mode
		) + user_input
		
		with st.chat_message( 'assistant' ):
			try:
				with st.spinner( 'Analyzing Guidance and Data...' ):
					response = client.responses.create(
						prompt={ 'id': PROMPT_ID, 'version': PROMPT_VERSION },
						input=[ { 'role': 'user', "content": full_input } ],
						include=[ "web_search_call.action.sources", "code_interpreter_call.outputs",
						],
					)
				
				answer = extract_answer( response )
				sources = extract_sources( response )
				analysis = extract_analysis( response )
				
				if answer:
					st.markdown( answer )
				elif sources or analysis[ 'tables' ]:
					st.info(
						'Response generated via tools. '
						'See Guidance or Analysis tabs.'
					)
				else:
					st.warning( 'No textual response returned.' )
				
				st.session_state.last_answer = answer
				st.session_state.last_sources = sources
				st.session_state.last_analysis = analysis
				
				st.session_state.chat_history.append(
					{
							'role': 'assistant',
							'content': answer or '' }
				)
			
			except Exception as e:
				st.error( 'An error occurred while processing the response.' )
				st.exception( e )

# =============================================================================
# GUIDANCE TAB
# =============================================================================
with tab_guidance:
	st.subheader( '' )
	if st.session_state.execution_mode == 'Analysis Only':
		st.info( 'Guidance Suppressed Due to Analysis Only mode.' )
	elif not st.session_state.last_sources:
		st.info( 'No guidance sources available.' )
	else:
		for i, src in enumerate( st.session_state.last_sources, start=1 ):
			st.markdown( f"**{i}. {src.get( 'title' ) or 'Source'}**" )
			if src.get( 'snippet' ):
				st.markdown( src[ 'snippet' ] )
			if src.get( 'url' ):
				st.markdown( f"[Link]({src[ 'url' ]})" )
			st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )

# =============================================================================
# ANALYSIS TAB
# =============================================================================
with tab_analysis:
	st.subheader( '' )
	if st.session_state.execution_mode == 'Guidance Only':
		st.info( 'Guidance Only mode.' )
	else:
		analysis = st.session_state.last_analysis
		
		if not (analysis[ 'tables' ] or analysis[ 'files' ] or analysis[ 'text' ]):
			st.info( 'No analysis artifacts available.' )
		else:
			for tbl in analysis[ 'tables' ]:
				if 'data' in tbl:
					st.dataframe( tbl[ 'data' ] )
			
			for txt in analysis[ 'text' ]:
				st.code( txt, language='text' )
			
			for f in analysis[ 'files' ]:
				file_id = f.get( 'id' )
				file_name = f.get( 'name', 'download' )
				if file_id:
					file_bytes = client.files.content( file_id ).read( )
					st.download_button( label=f'Download {file_name}', data=file_bytes,
						file_name=file_name, )

# ==============================================================================
# SYSTEM TAB
# ==============================================================================
with tab_system:
	st.subheader( '' )
	
	# ------------------
	# Prompt selector
	df_prompts = fetch_prompts_df( )
	prompt_names = [ '' ] + df_prompts[ 'Name' ].tolist( )
	
	selected_name: str = st.selectbox( 'Load System Prompt', prompt_names,
		key='system_prompt_selector' )
	
	st.session_state.pending_system_prompt_name = ( selected_name if selected_name else None )
	
	# ---------------
	# Action buttons
	col_load, col_clear, col_edit, col_spacer, col_xml_md, col_md_xml = st.columns( [ 1,
	                                                                                  1,
	                                                                                  1,
	                                                                                  0.5,
	                                                                                  1.5,
	                                                                                  1.5 ] )
	
	with col_load:
		load_clicked: bool = st.button( 'Load',
			disabled=st.session_state.pending_system_prompt_name is None )
	
	with col_clear:
		clear_clicked: bool = st.button( 'Clear' )
	
	with col_edit:
		edit_clicked: bool = st.button( 'Edit',
			disabled=st.session_state.pending_system_prompt_name is None )
	
	with col_spacer:
		st.write( '' )
	
	with col_xml_md:
		to_markdown_clicked: bool = st.button( 'XML → Markdown',
			help='Replace XML-like delimiters with Markdown headings (##).' )
	
	with col_md_xml:
		to_xml_clicked: bool = st.button( 'Markdown → XML',
			help='Replace Markdown headings (##) with XML-like delimiters.' )
	
	# -----------------
	# Button behaviors
	if load_clicked:
		record = fetch_prompt_by_name( st.session_state.pending_system_prompt_name )
		if record:
			st.session_state.system_prompt = record[ 'Text' ]
			st.session_state.selected_prompt_id = record[ 'PromptsId' ]
	
	if clear_clicked:
		st.session_state.system_prompt = ''
		st.session_state.selected_prompt_id = None
	
	if edit_clicked:
		record = fetch_prompt_by_name( st.session_state.pending_system_prompt_name )
		if record:
			st.session_state.selected_prompt_id = record[ 'PromptsId' ]
	
	if to_markdown_clicked:
		try:
			st.session_state.system_prompt = xml_converter(
				st.session_state.system_prompt
			)
			st.success( 'Converted to Markdown.' )
		except Exception as exc:
			st.error( f'Conversion failed: {exc}' )
	
	if to_xml_clicked:
		try:
			st.session_state.system_prompt = markdown_converter(
				st.session_state.system_prompt
			)
			st.success( 'Converted to XML-delimited format.' )
		except Exception as exc:
			st.error( f'Conversion failed: {exc}' )
	
	# ----------------------
	# System prompt editor
	st.text_area( 'System Prompt', key='system_prompt', height=260,
		help=( 'Edit System Instructions here. Use XML-like tags or Markdown headings (##). '
				'Conversion tools are provided above.' ) )

# ==============================================================================
# RETREIVAL TAB
# ==============================================================================
with tab_basic:
	st.subheader( '' )
	files = st.file_uploader( 'Upload documents', accept_multiple_files=True )
	if files:
		st.session_state.basic_docs.clear( )
		for f in files:
			st.session_state.basic_docs.extend( chunk_text( f.read( ).decode( errors='ignore' ) ) )
		st.success( f'{len( st.session_state.basic_docs )} chunks loaded' )

# ==============================================================================
# SEMANTIC SEARCH TAB
# ==============================================================================
with tab_semantic:
	st.subheader( '' )
	st.session_state.use_semantic = st.checkbox( 'Use Semantic Context',
		st.session_state.use_semantic )
	files = st.file_uploader( 'Upload for embedding', accept_multiple_files=True )
	if files:
		chunks = [ ]
		for f in files:
			chunks.extend( chunk_text( f.read( ).decode( errors='ignore' ) ) )
		vecs = embedder.encode( chunks )
		with sqlite3.connect( DB_PATH ) as conn:
			conn.execute( 'DELETE FROM embeddings' )
			for c, v in zip( chunks, vecs ):
				conn.execute(
					'INSERT INTO embeddings (chunk, vector) VALUES (?, ?)',
					(c, v.tobytes( )) )
		st.success( 'Semantic index built' )

# ==============================================================================
# PROMPT ENGINEERING TAB
# ==============================================================================
with tab_prompt:
	st.subheader( '' )
	df = fetch_prompts_df( )
	if st.session_state.selected_prompt_id:
		df[ 'Selected' ] = df[ 'PromptsId' ] == st.session_state.selected_prompt_id
	
	edited = st.data_editor( df, hide_index=True, use_container_width=True )
	sel = edited.loc[ edited[ 'Selected' ], 'PromptsId' ].tolist( )
	st.session_state.selected_prompt_id = sel[ 0 ] if len( sel ) == 1 else None
	
	prompt = fetch_prompt_by_id( st.session_state.selected_prompt_id ) or \
	         {  'Name': '', 'Text': '', 'Version': '', 'ID': '' }
	
	st.markdown( BLUE_DIVIDER, unsafe_allow_html=True )
	
	c1, c2 = st.columns( 2 )
	with c1:
		if st.button( '+ New' ):
			st.session_state.selected_prompt_id = None
			prompt = {
					'Name': '',
					'Text': '',
					'Version': '',
					'ID': '' }
	with c2:
		if st.button( '🗑 Delete', disabled=st.session_state.selected_prompt_id is None ):
			delete_prompt( st.session_state.selected_prompt_id )
			st.session_state.selected_prompt_id = None
			st.rerun( )
	
	name = st.text_input( 'Name', prompt[ 'Name' ] )
	version = st.text_input( 'Version', prompt[ 'Version' ] )
	pid = st.text_input( 'ID', prompt[ 'ID' ] )
	text = st.text_area( 'Prompt Text', prompt[ 'Text' ], height=260 )
	
	if st.button( '💾 Save' ):
		data = {
				'Name': name,
				'Text': text,
				'Version': version,
				'ID': pid }
		if st.session_state.selected_prompt_id:
			update_prompt( st.session_state.selected_prompt_id, data )
		else:
			insert_prompt( data )
		st.rerun( )

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

