'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                grok.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="grok.py" company="Terry D. Eppler">

	     grok.py
	     Copyright ©  2024  Terry Eppler

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

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    Groq Cloud API wrapper for Streamlit with Hybrid Tool support.
  </summary>
  ******************************************************************************************
'''

import os
import base64
import requests
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google.genai.types import ListFilesResponse
from xai_sdk.aio.image import ImageResponse

import config as cfg
from boogr import ErrorDialog, Error
import config as cfg
from xai_sdk import Client
from xai_sdk.chat import user, system, image, file

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encodes a local image to a base64 string for vision API requests."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Grok:
	"""
	
		Purpose:
		--------
		Base class for xAI (Grok) REST API functionality.
	
		This class provides:
			- API key and base URL management
			- Common request headers
			- Shared HTTP helpers for JSON and streaming requests
	
		Notes:
		------
		xAI exposes an OpenAI-compatible REST surface at:
			https://api.x.ai/v1
	
		All child capability classes (Chat, Images, Embeddings, Files, etc.)
		are expected to route through the helpers defined here.
	
	"""
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	organization: Optional[ str ]
	timeout: Optional[ float ]
	number: Optional[ int ]
	model: Optional[ str ]
	store: Optional[ bool ]
	response_format: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_output_tokens: Optional[ int ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	tool_choice: Optional[ str ]
	stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""
			
			Purpose:
			--------
			Initialize the Grok (xAI) API client.
	
			Parameters:
			-----------
			cfg : object
				Configuration object providing API credentials and options.
			
		"""
		self.api_key = cfg.XAI_API_KEY
		self.organization = None
		self.timeout = None
		self.instructions = None
		self.prompt = None
		self.store = None
		self.model = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.tool_choice = None
		self.response_format = None
		self.collections = None
		self.files = None

class Chat( Grok ):
	"""
	
		Purpose:
		--------
		Generate text and manage stateful conversations using the
		xAI Responses API.

		This class is a direct, faithful mapping of xAI's documented
		Responses API and does not emulate OpenAI legacy endpoints.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	user: Optional[ str ]
	model: Optional[ str ]
	reasoning_effort: Optional[ str ]
	previous_response_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	client: Optional[ Client ]
	chat_response: Optional[ Any  ]
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Chat capability and HTTP client.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.client = None
		self.prompt = None
		self.model = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.reasoning_effort = None
		self.previous_response_id = None
		self.tool_choice = 'auto'
		self.include = None
		self.collections = \
		{
				'Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee',
				'Financial Data': 'collection_3b4d5d26-d26f-487c-b589-1c5fbde26c5e'
		}
		self.files = \
		{
				'Outlays.csv': 'file_9d0acf02-4794-4a26-843b-b46c754e7cf5',
				'Authority.csv': 'file_b2b0139f-ceb1-491e-90e6-65d16152c521',
				'SF133.csv': 'file_41037cc2-e1f4-4cce-b25a-5c1d1f0172b2',
				'Account Balances.csv': 'file_41037cc2-e1f4-4cce-b25a-5c1d1f0172b2'
		}
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI text-capable models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'text', 'json_object' ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI text-capable models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4',
		         'grok-4-0709',
		         'grok-4-latest',
		         'grok-4-1-fast',
		         'grok-4-1-fast-reasoning',
		         'grok-4-1-fast-reasoning-latest',
		         'grok-4-1-fast-non-reasoning',
		         'grok-4-1-fast-non-reasoning-latest',
		         'grok-4-fast',
		         'grok-4-fast-reasoning',
		         'grok-4-fast-reasoning-latest',
		         'grok-4-fast-non-reasoning',
		         'grok-4-fast-non-reasoning-latest',
		         'grok-code-fast-1',
		         'grok-3',
		         'grok-3-latest',
		         'grok-3-mini',
		         'grok-3-fast',
		         'grok-3-fast-latest',
		         'grok-3-mini-fast',
		         'grok-3-mini-fast-latest' ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported reasoning effort levels.

			Notes:
			------
			Only valid for model = 'grok-3-mini'.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'low', 'high' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		return [ 'web_search_call_output',
		         'x_search_call_output',
		         'code_execution_call_output',
		         'collections_search_call_output',
		         'attachment_search_call_output',
		         'mcp_call_output',
		         'inline_citations',
		         'verbose_streaming' ]
	
	def create( self, prompt: str, model: str='grok-3-mini', max_tokens: int=10000,
			temperature: float=0.8, top_p: float=0.9, effort: str='high', format: str='text',
			store: bool=True, include: List[ str ]=None, instruct: str=None ):
		"""
		
			Purpose:
			--------
			Generate text using the xAI Responses API.

			If previous_response_id is set, the conversation will be
			continued server-side.

			Parameters:
			-----------
			prompt : str
				User input prompt.
			model : str | None
				Model identifier.
			max_output_tokens : int | None
				Maximum number of tokens in the response.
			temperature : float | None
			top_p : float | None
			include_reasoning : bool | None
				Whether to include encrypted reasoning content.
			reasoning_effort : str | None
				Reasoning effort level (grok-3-mini only).

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.max_output_tokens = max_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.instructions = instruct
			self.reasoning_effort = effort
			self.store = store
			self.response_format = format
			self.include = include
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {self.api_key}',
			                              'Content-Type': 'application/json', } )
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.user ) )
			chat_response = self.client.chat.create( model=self.model, messages=self.messages,
				store_messages=self.store, temperature=self.temperature, top_p=self.top_p, 
				reasoning_effort=self.reasoning_effort, max_tokens=self.max_output_tokens,
				response_format=self.response_format )
			return chat_response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Chat'
			ex.method = 'create( prompt: str, model: str )'
			error = ErrorDialog( ex )
			error.show( )

class Images( Grok ):
	"""
	
		Purpose:
		--------
		Provide image generation and image editing functionality using
		the xAI Images REST API.

		This class models the /images/generations and /images/edits
		endpoints exactly as exposed by xAI.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	model: Optional[ str ]
	aspect_ratio: Optional[ str ]
	resolution: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ Client ]
	image: Optional[ image ]
	image_path: Optional[ str ]
	detail: Optional[ str ]
	response_format: Optional[ str ]
	response: Optional[ ImageResponse ]
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Images API client.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.client = None
		self.model = None
		self.aspect_ratio = None
		self.resolution = None
		self.quality = None
		self.detail = None
		self.response_format = None
		self.client = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI image generation models.

			Returns:
			--------
			List[str]
		
		"""
		return [ "grok-2-image-1212", 'grok-imagine-image' ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		return [ '1:1',
		         '3:4',
		         '4:3',
		         '9:16',
		         '16:9',
		         '2:3',
		         '3:2',
		         '9:19.5',
		         '19.5:9',
		         '9:20',
		         '20:9',
		         '1:2',
		         '2:1']
	
	@property
	def resolution_options( self ) -> List[ str ]:
		return [ "1K",
		         "2K" ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		return [ "low",
		         "medium",
		         "high" ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		return [ "auto",
		         "low",
		         "high" ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'base64', 'url' ]
	
	def create( self, prompt: str, model: str='grok-imagine-image', resolution: str='1k',
			aspect_ratio: str='4:3',  format: str='base64' ) -> str | None:
		"""
		
			Purpose:
			--------
			Generate one or more images from a text prompt.

			Parameters:
			-----------
			prompt : str
			model : str | None
			n : int | None
			aspect_ratio : str | None
			resolution : str | None
			quality : str | None
			style : str | None
			response_format : str | None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = model
			self.resolution = resolution
			self.aspect_ratio = aspect_ratio
			self.response_format = format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update({ 'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', } )
			self.response = self.client.image.sample( prompt=self.prompt, resolution=self.resolution,
				model="grok-imagine-image", aspect_ratio=self.aspect_ratio,
				image_format=self.response_format )
			return self.response.base64
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'create( prompt: str, model: str )'
			error = ErrorDialog( ex )
			error.show( )
	
	def edit( self, image_path: str, prompt: str, model: str='grok-imagine-image',
			aspect_ratio: str = "4:3", resolution: str='1k', quality: str='medium',
			response_format: str='base64' ) -> str | None:
		"""
		
			Purpose:
			--------
			Edit an existing image using a text prompt and optional mask.

			Parameters:
			-----------
			image_path : str
			prompt : str
			mask_path : str | None
			model : str | None
			n : int | None
			aspect_ratio : str | None
			resolution : str | None
			quality : str | None
			style : str | None
			response_format : str | None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			throw_if( 'image_path', image_path )
			throw_if( 'prompt', prompt )
			self.model = model
			self.image_path = image_path
			self.aspect_ratio = aspect_ratio
			self.resolution = resolution
			self.quality = quality
			self.response_format = response_format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			with open( self.image_path, "rb" ) as f:
				image_data = base64.b64encode( f.read( ) ).decode( "utf-8" )
				self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
					aspect_ratio=self.aspect_ratio, image_format=self.response_format,
					image_url=f"data:image/jpeg;base64,{image_data}", )
				return self.response.base64
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def analyze( self, prompt: str, image_url: str, model: str='grok-4-1-fast-reasoning',
			max_output_tokens: int=10000, temperature: float=0.9, top_p: float=0.8,
			reasoning_effort: str='medium', detail: str='medium'  ):
		"""
		
			Purpose:
			--------
			Analyze an image (image understanding) using a text prompt and an image URL.

			This method uses xAI's multimodal input format via the Responses API and
			returns a text response describing or reasoning about the image.

			Parameters:
			-----------
			prompt : str
			image_url : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			include_reasoning : bool | None
			reasoning_effort : str | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( "prompt", prompt )
			throw_if( "image_url", image_url )
			self.model = model
			self.prompt = prompt
			self.image_url = image_url
			self.max_output_tokens = max_output_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.detail = detail
			self.reasoning_effort = reasoning_effort
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', } )
			chat_response = self.client.chat.create( model=self.model )
			chat_response.append( user( self.prompt,
				image( image_url=self.image_url, detail=self.detail ) ) )
			image_respose = chat_response.sample()
			return image_respose.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'analyze( prompt: str, image_url: str  )'
			error = ErrorDialog( ex )
			error.show( )

class Files( Grok ):
	"""
	
		Purpose:
		--------
		Provide file upload, retrieval, listing, deletion, and
		file-based querying functionality using the xAI (Grok) REST API.

		This class manages file storage and enables file-based chat
		via the Responses API.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	client: Optional[ Client ]
	prompt: Optional[ str ]
	file_name: Optional[ str ]
	response_format: Optional[ str ]
	instructions: Optional[ str ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	purpose: Optional[ str ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	file_ids: Optional[ List[ str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Files capability.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.client = None
		self.model = None
		self.content = None
		self.prompt = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.file_Name = None
		self.input = None
		self.purpose = None
		self.documents = \
		{
				'Account Balances.csv': 'file_9e0d8f9e-06c7-4495-9576-cdd83c433be6',
				'SF133.csv': 'file_41037cc2-e1f4-4cce-b25a-5c1d1f0172b2',
				'Authority.csv': '',
				'Outlays.csv': 'file_9d0acf02-4794-4a26-843b-b46c754e7cf5'
		}

	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return list of efficient file interaction models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-4-fast', 'grok-4' ]
	
	def upload( self, filepath: str, filename: str ):
		"""
		
			Purpose:
			--------
			Upload a local file to xAI file storage.

			Parameters:
			-----------
			file_path : str
			purpose : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'filename', filename )
			self.file_path = filepath
			self.file_name = filename
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
				'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.upload( open( self.file_path, mode='rb' ),
				filename=self.file_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'create( text: str )'
			error = ErrorDialog( ex )
			error.show( )
	
	def list( self ) -> ListFilesResponse:
		"""
		
			Purpose:
			--------
			List all stored files.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[dict]
		
		"""
		try:
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			files_response = self.client.files.list( )
			return files_response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list()'
			error = ErrorDialog( ex )
			error.show( )
	
	def retrieve( self, file_id: str ):
		"""
		
			Purpose:
			--------
			Retrieve metadata for a specific file.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.get( file_id=self.file_id )
			return self.file
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def extract( self, file_id: str ) -> bytes | None:
		"""
		
			Purpose:
			--------
			Retrieve raw content of a stored file.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			bytes
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			_content = self.client.files.content( file_id=self.file_id )
			return _content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def delete( self, file_id: str ):
		"""
		
			Purpose:
			--------
			Delete a file from storage.

			Parameters:
			-----------
			file_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.delete( file_id=self.file_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def search( self, filepath: str, filename: str, prompt: str, model: str='grok-4-fast' ) -> str | None:
		"""
		
			Purpose:
			--------
			Chat with an uploaded file by attaching it to a Responses API
			request and asking a question about its contents.

			Parameters:
			-----------
			file_id : str
			prompt : str
			model : str | None
			max_output_tokens : int | None
			temperature : float | None
			top_p : float | None
			store : bool
			previous_response_id : str | None

			Returns:
			--------
			str
		
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'filename', filename )
			throw_if( 'prompt', prompt )
			self.model = model
			self.prompt = prompt
			self.file_path = filepath
			self.filename = filename
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( {
					'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.file = self.client.files.upload( open( self.file_path, 'rb' ),
				filename=self.file_name )
			chat_response = self.client.chat.create( model=self.model )
			chat_response.append( user( self.prompt, file( self.file.id ) ) )
			_response = chat_response.sample()
			return _response.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'documents',
		         'response',
		         'name',
		         'model',
		         'file_id',
		         'list',
		         'retrieve',
		         'search',
		         'delete',
		         'upload', ]

class VectorStores( Grok ):
	"""
	
		Purpose:
		--------
		Provide access to xAI Collections for grouping uploaded documents
		and reusing them across Responses-based interactions.

		This class manages collection metadata and membership only.
		Collections are referenced by ID in other APIs (e.g. Responses).

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	
	"""
	client: Optional[ Client ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	content: Optional[ str ]
	name: Optional[ str ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	store_id: Optional[ str ]
	document: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.client = None
		self.model = None
		self.content = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.file_name = None
		self.store_id = None
		self.document = None
		self.client.headers.update( {'Authorization': f'Bearer {cfg.GROK_API_KEY}',
		                             'Content-Type': 'application/json', } )
		self.collections = \
		{
				'Financial Data': 'collection_3b4d5d26-d26f-487c-b589-1c5fbde26c5e',
				'DoD Data': 'collection_137a5ed3-2f20-4082-bf44-73df43a356a4',
				'DoD Regulations': 'collection_a7973fd2-a336-4ed0-a495-4ffa947041c6',
				'DoA Regulations': 'collection_dbf8919e-5f56-435b-806b-642cd57c355e',
				'Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee',
		}
	
	def list( self ):
		"""
		
			Purpose:
			--------
			List all collections accessible to the account.

			Returns:
			--------
			List[dict]
		
		"""
		try:
			self.client = Client( api_key=self.api_key )
			self.collections = client.collections.list( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = ''
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def retrieve( self, collection_id: str ):
		"""
		
			Purpose:
			--------
			Retrieve metadata for a specific collection.

			Parameters:
			-----------
			collection_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'collection_id', collection_id )
			
			self.collection_id = collection_id
			
			url = f'{self.base_url}/collections/{self.collection_id}'
			response = self.client.get( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = ''
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def create( self, name: str, file_ids: List[ str ], description: Optional[ str ] = None ):
		"""
		
			Purpose:
			--------
			Create a new collection with an initial set of files.

			Parameters:
			-----------
			name : str
			file_ids : List[str]
			description : str | None

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'name', name )
			throw_if( 'file_ids', file_ids )
			
			payload = {
					'name': name,
					'file_ids': file_ids }
			if description:
				payload[ 'description' ] = description
			
			url = f'{self.base_url}/collections'
			response = self.client.post( url, json=payload, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = ''
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def search( self, prompt: str, store_id: str, model: str = 'grok-4-fast' ) -> str | None:
		"""

	        Purpose:
	        _______
	        Method that analyzeses an image given a prompt,

	        Parameters:
	        ----------
	        prompt: str
	        url: str

	        Returns:
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			self.prompt = prompt
			self.model = model
			self.store_id = store_id
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.response = client.collections.search( query=self.prompt,
				collection_ids=[ self.store_id ],)
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def update( self, collection_id: str, filepath: str, filename: stre ):
		"""
		
			Purpose:
			--------
			Update collection membership by adding or removing files.

			Parameters:
			-----------
			collection_id : str
			add_file_ids : List[str] | None
			remove_file_ids : List[str] | None

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'collection_id', collection_id )
			throw_if( 'filename', filename )
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.file_name = filename
			self.store_id = collection_id
			with open( self.file_path, 'rb' ) as file:
				file_data = file.read( )
				self.document = client.collections.upload_document( collection_id=self.store_id,
					name=self.file_name, data=file_data, content_type="text/html", )
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = ''
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
			
	def delete( self, collection_id: str ):
		"""
		
			Purpose:
			--------
			Delete a collection.

			Parameters:
			-----------
			collection_id : str

			Returns:
			--------
			dict
		
		"""
		try:
			throw_if( 'collection_id', collection_id )
			
			url = f'{self.base_url}/collections/{collection_id}'
			response = self.client.delete( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = ''
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response',
		         'name',
		         'model',
		         'file_id',
		         'store_id',
		         'create',
		         'retrieve',
		         'search',
		         'delete',
		         'update', ]
	