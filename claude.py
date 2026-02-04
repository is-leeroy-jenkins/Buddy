'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                claude.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="claude.py" company="Terry D. Eppler">

	     claude.py
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
    claude.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
from boogr import Error, ErrorDialog
import config as cfg

# ======================================================================================
# Utility
# ======================================================================================

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

# ======================================================================================
# Base Class — Claude
# ======================================================================================

class Claude:
	"""
	Purpose:
	--------
	Base class encapsulating shared configuration and HTTP behavior for Anthropic Claude.

	Attributes:
	-----------
	api_key : str
		Anthropic API key.
	base_url : str
		Claude API base URL.
	headers : Dict[str, str]
		Default HTTP headers.
	response : Optional[Dict[str, Any]]
		Raw API response payload.
	"""
	
	api_key: str
	base_url: str
	headers: Dict[ str, str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
		Purpose:
		--------
		Initialize shared Claude configuration.

		Returns:
		--------
		None
		"""
		self.api_key = cfg.CLAUDE_API_KEY
		self.base_url = 'https://api.anthropic.com/v1'
		self.headers = {
				'Content-Type': 'application/json',
				'x-api-key': self.api_key,
				'anthropic-version': '2023-06-01',
		}
		self.response = None

# ======================================================================================
# Chat — Text / Reasoning
# ======================================================================================

class Chat( Claude ):
	"""
	Purpose:
	--------
	Provide text-based chat and reasoning using Claude models.

	Attributes:
	-----------
	model : Optional[str]
		Claude model name.
	system_prompt : Optional[str]
		Optional system instruction.
	"""
	
	model: Optional[ str ]
	system_prompt: Optional[ str ]
	
	def __init__( self, system_prompt: Optional[ str ] = None ) -> None:
		"""
		Purpose:
		--------
		Initialize a Chat instance.

		Parameters:
		-----------
		system_prompt : Optional[str]
			Optional system-level instruction.

		Returns:
		--------
		None
		"""
		super( ).__init__( )
		self.system_prompt = system_prompt
		self.model = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		Purpose:
		--------
		List supported Claude chat models.

		Returns:
		--------
		List[str]
		"""
		return [
				'claude-3-opus-20240229',
				'claude-3-sonnet-20240229',
				'claude-3-haiku-20240307',
		]
	
	def generate_text( self, prompt: str,
			model: str = 'claude-3-sonnet-20240229',
			max_tokens: int = 4096 ) -> str | None:
		"""
		Purpose:
		--------
		Generate a text response using Claude.

		Parameters:
		-----------
		prompt : str
			User input text.
		model : str
			Claude model name.
		max_tokens : int
			Maximum tokens to generate.

		Returns:
		--------
		str | None
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = model
			
			payload: Dict[ str, Any ] = {
					'model': self.model,
					'max_tokens': max_tokens,
					'messages': [
							{
									'role': 'user',
									'content': prompt,
							}
					],
			}
			
			if self.system_prompt:
				payload[ 'system' ] = self.system_prompt
			
			response = requests.post(
				url=f'{self.base_url}/messages',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'content' ][ 0 ][ 'text' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'claude'
			exception.cause = 'Chat'
			exception.method = 'generate_text(self, prompt, model, max_tokens)'
			ErrorDialog( exception ).show( )
			return None

# ======================================================================================
# Vision — Image Analysis
# ======================================================================================

class Vision( Claude ):
	"""
	Purpose:
	--------
	Provide multimodal image understanding using Claude vision-capable models.
	"""
	
	def analyze_image( self, prompt: str,
			image_url: str,
			model: str = 'claude-3-sonnet-20240229',
			max_tokens: int = 4096 ) -> str | None:
		"""
		Purpose:
		--------
		Analyze an image with a textual instruction.

		Parameters:
		-----------
		prompt : str
			Instruction or question about the image.
		image_url : str
			URL of the image.
		model : str
			Claude model name.
		max_tokens : int
			Maximum tokens to generate.

		Returns:
		--------
		str | None
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'image_url', image_url )
			
			payload: Dict[ str, Any ] = {
					'model': model,
					'max_tokens': max_tokens,
					'messages': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'text',
													'text': prompt },
											{
													'type': 'image_url',
													'image_url': image_url },
									],
							}
					],
			}
			
			response = requests.post(
				url=f'{self.base_url}/messages',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'content' ][ 0 ][ 'text' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'claude'
			exception.cause = 'Vision'
			exception.method = 'analyze_image(self, prompt, image_url, model, max_tokens)'
			ErrorDialog( exception ).show( )
			return None

# ======================================================================================
# Document — PDF / Document Ingestion
# ======================================================================================

class Document( Claude ):
	"""
	Purpose:
	--------
	Analyze PDFs and document files using Claude's native document support.
	"""
	
	def analyze_pdf( self, file_id: str,
			prompt: str,
			model: str = 'claude-3-sonnet-20240229',
			max_tokens: int = 4096 ) -> str | None:
		"""
			Purpose:
			--------
			Analyze a PDF using a previously uploaded file_id.
	
			Parameters:
			-----------
			file_id : str
				Claude Files API file identifier.
			prompt : str
				Instruction for document analysis.
			model : str
				Claude model name.
			max_tokens : int
				Maximum tokens to generate.
	
			Returns:
			--------
			str | None
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'prompt', prompt )
			
			payload: Dict[ str, Any ] = {
					'model': model,
					'max_tokens': max_tokens,
					'messages': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'document',
													'source': {
															'type': 'file_id',
															'file_id': file_id,
													},
											},
											{
													'type': 'text',
													'text': prompt,
											},
									],
							}
					],
			}
			
			response = requests.post(
				url=f'{self.base_url}/messages',
				headers=self.headers,
				json=payload,
				timeout=120,
			)
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'content' ][ 0 ][ 'text' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'claude'
			exception.cause = 'Document'
			exception.method = 'analyze_pdf(self, file_id, prompt, model, max_tokens)'
			ErrorDialog( exception ).show( )