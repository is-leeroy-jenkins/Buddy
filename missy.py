'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                missy.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="missy.py" company="Terry D. Eppler">

	     missy.py
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
    missy.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
from boogr import Error, ErrorDialog
import config as cfg


def throw_if( name: str, value: object ) -> None:
	"""
	
		Purpose:
		--------
		Guard clause that raises a ValueError if a required argument is None.
	
		Parameters:
		-----------
		name : str
			Name of the argument being validated.
		value : object
			Value to validate.
	
		Returns:
		--------
		None
	
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Mistral( ):
	"""
	
		Purpose:
		--------
		Base class encapsulating shared configuration, sampling controls, headers, and
		HTTP client behavior for Mistral-based LLM interactions.
	
		Attributes:
		-----------
		api_key : str
			API key used to authenticate with the Mistral API.
		base_url : str
			Root URL for the Mistral API.
		headers : Dict[str, str]
			Default HTTP headers for requests.
		temperature : Optional[float]
			Sampling temperature.
		top_p : Optional[float]
			Nucleus sampling parameter.
		max_tokens : Optional[int]
			Maximum number of tokens to generate.
		stream : Optional[bool]
			Whether to request streamed responses.
		prompt : Optional[str]
			Current prompt being processed.
		response : Optional[Dict[str, Any]]
			Raw response payload from the API.
			
	"""
	api_key: str
	base_url: str
	headers: Dict[ str, str ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	max_tokens: Optional[ int ]
	stream: Optional[ bool ]
	prompt: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize shared configuration for Mistral API usage.
	
			Returns:
			--------
			None
			
		"""
		self.api_key = cfg.MISTRAL_API_KEY
		self.base_url = 'https://api.mistral.ai/v1'
		self.headers = { 'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}', }
		self.temperature = None
		self.top_p = None
		self.max_tokens = None
		self.stream = None
		self.prompt = None
		self.response = None

class Chat( Mistral ):
	"""
	
		Purpose:
		--------
		Class providing chat and text-generation capabilities using Mistral chat models.
		This mirrors the Chat(GPT) interface and usage patterns found in gpt.py.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral model to use.
		messages : Optional[List[Dict[str, Any]]]
			Message payload sent to the chat completion endpoint.
	
		Methods:
		--------
		generate_text(prompt, model, temperature, top_p, max_tokens, stream)
			Generate a text or chat completion.
			
	"""
	model: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self, temperature: float=0.8, top_p: float=0.9, max_tokens: int=8192,
			stream: bool=False ) -> None:
		"""
			
			Purpose:
			--------
			Initialize a Chat instance with default sampling parameters.
	
			Parameters:
			-----------
			temperature : float
				Sampling temperature.
			top_p : float
				Nucleus sampling value.
			max_tokens : int
				Maximum number of tokens to generate.
			stream : bool
				Whether to request streaming output.
	
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.temperature = temperature
		self.top_p = top_p
		self.max_tokens = max_tokens
		self.stream = stream
		self.model = None
		self.messages = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			List supported Mistral chat models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest',
				'open-mixtral-8x7b', 'open-mixtral-8x22b', ]
	
	def generate_text( self, prompt: str, model: str='mistral-small-latest',
			temperature: float=None, top_p: float=None, max_tokens: int=None,
			stream: bool=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Generate a chat completion given a prompt using the Mistral API.
	
			Parameters:
			-----------
			prompt : str
				Input text or user message.
			model : str
				Name of the Mistral model to use.
			temperature : Optional[float]
				Sampling temperature override.
			top_p : Optional[float]
				Nucleus sampling override.
			max_tokens : Optional[int]
				Maximum token override.
			stream : Optional[bool]
				Streaming override.
	
			Returns:
			--------
			str | None
				Generated text output.
				
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			payload = { 'model': self.model,
			            'messages': [ { 'role': 'user', 'content': self.prompt, } ],
					'temperature': temperature if temperature is not None else self.temperature,
					'top_p': top_p if top_p is not None else self.top_p,
					'max_tokens': max_tokens if max_tokens is not None else self.max_tokens,
					'stream': stream if stream is not None else self.stream,
			}
			
			response = requests.post(
				url=f'{self.base_url}/chat/completions',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Chat'
			exception.method = 'generate_text(self, prompt, model, temperature, top_p, max_tokens, stream)'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Return a list of public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
			
		"""
		return [
				'temperature',
				'top_p',
				'max_tokens',
				'stream',
				'prompt',
				'response',
				'model',
				'model_options',
				'generate_text',
		]


class Embedding( Mistral ):
	"""
		
		Purpose:
		--------
		Class providing vector embedding generation using Mistral embedding models. This mirrors
		the Embedding(GPT) interface and conventions found in gpt.py.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral embedding model to use.
		vectors : Optional[List[List[float]]]
			Generated embedding vectors.
	
		Methods:
		--------
		create(input_text, model)
			Generate embeddings for input text.
		
	"""
	model: Optional[ str ]
	vectors: Optional[ List[ List[ float ] ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize an Embedding instance for Mistral embedding generation.
	
			Returns:
			--------
			None
			
		"""
		super( ).__init__( )
		self.model = None
		self.vectors = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			List supported Mistral embedding models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-embed', ]
	
	def create( self, input_text: List[ str ], model: str='mistral-embed' ) -> List[ List[ float ] ] | None:
		"""
		
			Purpose:
			--------
			Generate embeddings for the provided input text using the Mistral API.
	
			Parameters:
			-----------
			input_text : str | List[str]
				Input text or list of text segments to embed.
			model : str
				Name of the Mistral embedding model to use.
	
			Returns:
			--------
			List[List[float]] | None
				List of embedding vectors.
				
		"""
		try:
			throw_if( 'input_text', input_text )
			self.model = model
			payload  = { 'model': self.model, 'input': input_text, }
			response = requests.post(
				url=f'{self.base_url}/embeddings',
				headers=self.headers,
				json=payload,
				timeout=120, )
			response.raise_for_status( )
			data = response.json( )
			self.vectors = [ item[ 'embedding' ] for item in data.get( 'data', [ ] ) ]
			return self.vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Embedding'
			exception.method = 'create(self, input_text, model)'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return a list of public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
			
		"""
		return [ 'model', 'model_options', 'vectors', 'create', ]

class Image( Mistral ):
	"""
	
		Purpose:
		--------
		Provide multimodal image understanding and analysis using Mistral vision-capable chat
		models. This class isolates image reasoning from standard text chat while continuing
		to use the Chat Completions API.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral vision-capable model.
		prompt : Optional[str]
			Instruction or question associated with the image.
		image_url : Optional[str]
			URL of the image to analyze.
		response : Optional[Dict[str, Any]]
			Raw response payload from the API.
			
	"""
	
	model: Optional[ str ]
	prompt: Optional[ str ]
	image_url: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize a Vision instance for multimodal image analysis.
	
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.model = None
		self.prompt = None
		self.image_url = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			List supported Mistral vision-capable models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-large-latest', 'mistral-large-2512', ]
	
	def analyze_image( self, prompt: str, image_url: str,
			model: str = 'mistral-large-latest',
			temperature: Optional[ float ] = None,
			top_p: Optional[ float ] = None,
			max_tokens: Optional[ int ] = None ) -> str | None:
		"""
			
			Purpose:
			--------
			Analyze an image using a multimodal prompt via the Mistral Chat Completions API.
	
			Parameters:
			-----------
			prompt : str
				Instruction or question describing the desired image analysis.
			image_url : str
				URL pointing to the image to analyze.
			model : str
				Vision-capable Mistral model name.
			temperature : Optional[float]
				Sampling temperature override.
			top_p : Optional[float]
				Nucleus sampling override.
			max_tokens : Optional[int]
				Maximum token override.
	
			Returns:
			--------
			str | None
				Generated image analysis output.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'image_url', image_url )
			
			self.prompt = prompt
			self.image_url = image_url
			self.model = model
			
			payload: Dict[ str, Any ] = {
					'model': self.model,
					'messages': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'text',
													'text': self.prompt,
											},
											{
													'type': 'image_url',
													'image_url': {
															'url': self.image_url,
													},
											},
									],
							}
					],
					'temperature': temperature if temperature is not None else self.temperature,
					'top_p': top_p if top_p is not None else self.top_p,
					'max_tokens': max_tokens if max_tokens is not None else self.max_tokens,
			}
			
			response = requests.post(
				url=f'{self.base_url}/chat/completions',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Vision'
			exception.method = (
					'analyze_image(self, prompt, image_url, model, temperature, top_p, max_tokens)'
			)
			error = ErrorDialog( exception )
			error.show( )
			return None
	
	def __dir__( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Return public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
		
		"""
		return [
				'model',
				'model_options',
				'prompt',
				'image_url',
				'response',
				'analyze_image',
		]


class Document( Mistral ):
	"""
	
		Purpose:
		--------
		Provide document-oriented reasoning, summarization, and structured extraction by
		applying Mistral language models to locally extracted document text.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral model used for document reasoning.
		document_text : Optional[str]
			Extracted textual content of the document.
		response : Optional[Dict[str, Any]]
			Raw response payload from the API.
		
	"""
	model: Optional[ str ]
	document_text: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
			
			Purpose:
			--------
			Initialize a Document instance for document-level reasoning.
	
			Returns:
			--------
			None
			
		"""
		super( ).__init__( )
		self.model = None
		self.document_text = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			List supported Mistral models appropriate for document analysis.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-large-latest', 'mistral-large-2512', ]
	
	def analyze_document( self, document_text: str, prompt: str,
			model: str = 'mistral-large-latest',
			temperature: Optional[ float ] = None,
			top_p: Optional[ float ] = None,
			max_tokens: Optional[ int ] = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Analyze extracted document text using a structured prompt and Mistral reasoning.
	
			Parameters:
			-----------
			document_text : str
				Extracted textual content of the document.
			prompt : str
				Instruction describing the desired analysis or extraction.
			model : str
				Mistral model name.
			temperature : Optional[float]
				Sampling temperature override.
			top_p : Optional[float]
				Nucleus sampling override.
			max_tokens : Optional[int]
				Maximum token override.
	
			Returns:
			--------
			str | None
				Generated document analysis output.
			
		"""
		try:
			throw_if( 'document_text', document_text )
			throw_if( 'prompt', prompt )
			
			self.document_text = document_text
			self.model = model
			
			payload: Dict[ str, Any ] = {
					'model': self.model,
					'messages': [
							{
									'role': 'user',
									'content': f'{prompt}\n\n{self.document_text}',
							}
					],
					'temperature': temperature if temperature is not None else self.temperature,
					'top_p': top_p if top_p is not None else self.top_p,
					'max_tokens': max_tokens if max_tokens is not None else self.max_tokens,
			}
			
			response = requests.post(
				url=f'{self.base_url}/chat/completions',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Document'
			exception.method = (
					'analyze_document(self, document_text, prompt, model, temperature, top_p, max_tokens)'
			)
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Return public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
		
		"""
		return [
				'model',
				'model_options',
				'document_text',
				'response',
				'analyze_document',
		]