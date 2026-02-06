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
import groq
from groq import Groq
from xai_sdk.aio.image import ImageResponse

import config as cfg
from boogr import ErrorDialog, Error
import config as cfg
from xai_sdk import Client
from xai_sdk.chat import user, system, image

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
	max_completion_tokens: Optional[ int ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	tool_choice: Optional[ str ]
	
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
		self.api_key = cfg.GROQ_API_KEY
		self.organization = None
		self.timeout = None
		self.instructions = None
		self.prompt = None
		self.store = None
		self.model = None
		self.max_tokens = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.tool_choice = None
		self.response_format = None

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
		self.model = None
		self.max_tokens = None
		self.temperature = None
		self.top_percent = None
		self.reasoning_effort = None
		self.previous_response_id = None
		self.tool_choice = 'auto'
		self.include = None
	
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
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.instructions = instruct
			self.reasoning_effort = effort
			self.store = store
			self.response_format = format
			self.include = include
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}',
			                              'Content-Type': 'application/json', } )
			self.messages.append( system( self.instructions ) )
			self.messages.append( user( self.prompt ) )
			chat_response = self.client.chat.create( model=self.model, messages=self.messages, 
				store_messages=self.store, temperature=self.temperature, top_p=self.top_p, 
				reasoning_effort=self.reasoning_effort, response_format=self.response_format)
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
	style: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ Client ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
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
		self.model = None
		self.aspect_ratio = None
		self.resolution = None
		self.quality = None
		self.style = None
		self.response_format = None
		self.client = None
	

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
	def aspect_ratio_options( self ) -> List[ str ]:
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
		return [ "standard",
		         "high" ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		return [ "low",
		         "medium",
		         "high" ]
	
	@property
	def style_options( self ) -> List[ str ]:
		return [ "natural",
		         "illustration",
		         "photorealistic" ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'base64', 'url' ]
	
	def create( self, prompt: str, url: str, model: str='grok-imagine-image', n: int=None,
			aspect_ratio: str=None, resolution: str=None, quality: str=None,
			style: str=None, format: str='base64' ) -> ImageResponse | None:
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
			throw_if( 'url', url )
			self.image_url = url
			self.model = model
			self.aspect_ratio = aspect_ratio
			self.resolution = resolution
			self.quality = quality
			self.style = style
			self.response_format = format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			self.response = self.client.image.sample( prompt=self.prompt,
				model="grok-imagine-image", aspect_ratio=self.aspect_ratio,
				image_format=self.response_format, image_url=self.image_url, )
			return self.response.image
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def edit( self, image_path: str, prompt: str, model: str='grok-imagine-image',
			aspect_ratio: str=None, resolution: str=None,
			quality: str=None, style: str=None, response_format: str=None ):
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
			self.style = style
			self.response_format = response_format
			self.client = Client( api_key=self.api_key )
			self.client.headers.update( { 'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )
			with open( self.image_path, "rb" ) as f:
				image_data = base64.b64encode( f.read( ) ).decode( "utf-8" )
				self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
					aspect_ratio=self.aspect_ratio, image_url=f"data:image/jpeg;base64,{image_data}", )
				return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def analyze( self, prompt: str, image_url: str, model: str=None, max_output_tokens: int=None,
			temperature: float=None, top_p: float=None, include_reasoning: bool=None,
			reasoning_effort: str=None, store: bool=False  ):
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
			self.max_output_tokens = max_output_tokens
			self.temperature = temperature
			self.top_p = top_p
			self.include_reasoning = include_reasoning
			self.reasoning_effort = reasoning_effort
			self.previous_response_id = previous_response_id
			if self.reasoning_effort and self.model != "grok-3-mini":
				raise ValueError( "reasoning_effort is only supported when model == 'grok-3-mini'." )
			
			payload = {
					"model": self.model,
					"store": store,
					"input": [ {
							           "role": "user",
							           "content": [ {
									                        "type": "input_text",
									                        "text": prompt },
							                        {
									                        "type": "input_image",
									                        "image_url": image_url } ] } ],
					"max_output_tokens": self.max_output_tokens,
					"temperature": self.temperature,
					"top_p": self.top_p }
			
			if self.previous_response_id:
				payload[ "previous_response_id" ] = self.previous_response_id
			
			if self.include_reasoning:
				payload[ "include" ] = [ "reasoning.encrypted_content" ]
			
			if self.reasoning_effort:
				payload[ "reasoning_effort" ] = self.reasoning_effort
			
			url = f"{self.base_url}/responses"
			response = self.client.post( url, json=payload, timeout=self.timeout )
			response.raise_for_status( )
			
			data = response.json( )
			self.previous_response_id = data.get( "id" )
			
			for output in data.get( "output", [ ] ):
				for content in output.get( "content", [ ] ):
					if content.get( "type" ) == "output_text":
						return content.get( "text", "" )
			
			return ""
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )


class Embeddings( Grok ):
	"""
	
		Purpose:
		--------
		Provide text embedding generation functionality using the
		xAI (Grok) REST API.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		None
	
	"""
	
	model: Optional[ str ]
	input_text: Optional[ str ]
	encoding_format: Optional[ str ]
	client: requests.Session
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Embeddings capability.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.model = None
		self.input_text = None
		self.encoding_format = None
		self.client = requests.Session( )
		self.client.headers.update({
					'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json',
			})
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported embedding-capable model identifiers.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]
		
		"""
		return [ 'grok-2-embedding' ]
	
	@property
	def encoding_format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported embedding output formats.
		
			Parameters:
			-----------
			None
		
			Returns:
			--------
			List[str]
		
		"""
		return [ 'float',
		         'base64' ]

	def create( self, text: str, model: str=None, encoding_format: str=None ):
		"""
		
			Purpose:
			--------
			Generate an embedding vector for input text.
		
			Parameters:
			-----------
			text : str
			model : str | None
			encoding_format : str | None
		
			Returns:
			--------
			list[float] | str
		
		"""
		try:
			throw_if( 'text', text )
			
			self.input_text = text
			self.model = model
			self.encoding_format = encoding_format
			
			payload = {
					'model': self.model,
					'input': self.input_text,
					'encoding_format': self.encoding_format,
			}
			
			url = f'{self.base_url}/embeddings'
			response = self.client.post( url, json=payload, timeout=self.timeout )
			response.raise_for_status( )
			
			data = response.json( )
			return data[ 'data' ][ 0 ][ 'embedding' ]
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
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
	
	file_id: Optional[ str ]
	purpose: Optional[ str ]
	client: requests.Session
	
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
		
		self.file_id = None
		self.purpose = None
		self.client = requests.Session( )
		self.client.headers.update( {'Authorization': f'Bearer {cfg.GROK_API_KEY}' } )

	@property
	def purpose_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported file purpose identifiers.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'vision',
		         'image_edit',
		         'responses',
		         'fine_tune' ]
	
	def upload( self, file_path: str, purpose: str ):
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
			throw_if( 'file_path', file_path )
			throw_if( 'purpose', purpose )
			
			self.purpose = purpose
			
			with open( file_path, 'rb' ) as fh:
				files = { 'file': fh }
				data = { 'purpose': self.purpose }
				
				url = f'{self.base_url}/files'
				response = self.client.post( url, files=files, data=data, timeout=self.timeout )
				response.raise_for_status( )
			
			return response.json( )
		
		except Exception as ex:
			raise ex
	
	def list( self ):
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
			url = f'{self.base_url}/files'
			response = self.client.get( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( ).get( 'data', [ ] )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
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
			
			url = f'{self.base_url}/files/{self.file_id}'
			response = self.client.get( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def retrieve_content( self, file_id: str ):
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
			
			url = f'{self.base_url}/files/{self.file_id}/content'
			response = self.client.get( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
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
			
			url = f'{self.base_url}/files/{self.file_id}'
			response = self.client.delete( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )
	
	def query( self, file_id: str, prompt: str, model: str=None, max_output_tokens: int=None,
			temperature: float=None, top_p: float=None, store: bool=True,
			previous_response_id: str=None ):
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
			throw_if( 'file_id', file_id )
			throw_if( 'prompt', prompt )
			
			payload = {
					'model': model,
					'store': store,
					'input': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'input_text',
													'text': prompt },
											{
													'type': 'input_file',
													'file_id': file_id },
									],
							}
					],
					'max_output_tokens': max_output_tokens,
					'temperature': temperature,
					'top_p': top_p,
			}
			
			if previous_response_id:
				payload[ 'previous_response_id' ] = previous_response_id
			
			url = f'{self.base_url}/responses'
			response = self.client.post( url, json=payload, timeout=self.timeout )
			response.raise_for_status( )
			
			data = response.json( )
			
			for output in data.get( 'output', [ ] ):
				for content in output.get( 'content', [ ] ):
					if content.get( 'type' ) == 'output_text':
						return content.get( 'text', '' )
			
			return ''
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Embeddings'
			ex.method = ''
			error = ErrorDialog( ex )
			error.show( )


class Collections( Grok ):
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
	
	collection_id: Optional[ str ]
	name: Optional[ str ]
	client: requests.Session
	
	def __init__( self ):
		"""
		
			Purpose:
			--------
			Initialize the Collections capability.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		
		self.collection_id = None
		self.name = None
		
		self.client = requests.Session( )
		self.client.headers.update(
			{
					'Authorization': f'Bearer {cfg.GROK_API_KEY}',
					'Content-Type': 'application/json',
			}
		)
	
	@property
	def known_collections( self ) -> Dict[ str, str ]:
		"""
		
			Purpose:
			--------
			Return known, pre-existing collections visible in the xAI dashboard.

			These collections already exist server-side and can be referenced
			directly in Responses requests.

			Returns:
			--------
			dict[str, str]
		
		"""
		return {
				'armory': 'collection_a7973fd2-a336-4ed0-0495-4ff947041c6',
				'doa_regulations': 'collection_dbf8919e-5f56-435b-806b-642cd57c355e',
				'financial_regulations': 'collection_9195847-03a1-443c-9240-294c64dd01e2',
				'explanatory_statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'public_laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee',
		}
	
	@property
	def collection_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return collection names for UI selection.

			Returns:
			--------
			List[str]
		
		"""
		return list( self.known_collections.keys( ) )
	
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
			url = f'{self.base_url}/collections'
			response = self.client.get( url, timeout=self.timeout )
			response.raise_for_status( )
			
			return response.json( ).get( 'data', [ ] )
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
	
	def update( self, collection_id: str, add_file_ids: Optional[
		List[ str ] ] = None, remove_file_ids: Optional[ List[ str ] ] = None ):
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
			
			payload = { }
			if add_file_ids:
				payload[ 'add_file_ids' ] = add_file_ids
			if remove_file_ids:
				payload[ 'remove_file_ids' ] = remove_file_ids
			
			url = f'{self.base_url}/collections/{collection_id}'
			response = self.client.patch( url, json=payload, timeout=self.timeout )
			response.raise_for_status( )
			
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
