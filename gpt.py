'''
	******************************************************************************************
	    Assembly:                Boo
	    Filename:                Boo.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2022
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="gpt.py" company="Terry D. Eppler">
	
	           Boo is a df analysis tool integrating various Generative GPT, GptText-Processing, and
	           Machine-Learning algorithms for federal analysts.
	           Copyright ©  2022  Terry Eppler
	
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
	  Boo.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
import os
from openai.pagination import SyncCursorPage
from pathlib import Path
import tiktoken
from openai import OpenAI
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	import openai.types.responses
	from openai.types import CreateEmbeddingResponse, VectorStore
	
from boogr import Error
import config as cfg

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class GPT:
	'''
	
	    Purpose:
	    --------
	    Base class for OpenAI functionality.

    '''
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_output_tokens: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	store: Optional[ bool ]
	
	def __init__( self ):
		self.api_key = cfg.OPENAI_API_KEY
		self.modalities = [  ]
		self.stops = [  ]
		self.response_format = None
		self.number = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_output_tokens = None
		self.prompt = None
		self.client = None
		self.store = None

class Chat( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's  ChatGPT API
	
	
	    Parameters
	    ------------
	    number: int=1
	    temperature: float=0.8
	    top_p: float=0.9
	    frequency: float=0.0
	    presence: float=0.0
	    max_tokens: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    number,
	    temperature,
	    top_percent,
	    frequency_penalty,
	    presence_penalty,
	    store,
	    stream,
	    maximum_completion_tokens,
	    api_key,
	    client,
	    model,
	    embedding,
	    response,
	    modalities,
	    stops,
	    content,
	    prompt,
	    response,
	    file_path,
	    path,
	    messages,
	    image_url,
	    response_format,
	    tools,
	    vector_store_ids
	    
	    Properties:
	    -----------
	    model_options - List[ str ]
	
	    Methods
	    ------------
	    generate_text( self, prompt: str ) -> str:
	    analyze_image( self, prompt: str, url: str ) -> str:
	    summarize_document( self, prompt: str, path: str ) -> str
	    search_web( self, prompt: str ) -> str
	    search_files( self, prompt: str ) -> str
	    dump( self ) -> str
	    get_data( self ) -> { }


    """
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	file_url: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ List[ Dict[ str, Any ] ] ]
	vector_stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ openai.types.responses.Response ]
	file: Optional[ openai.types.file_object.FileObject ]
	purpose: Optional[ str ]
	
	def __init__( self, model: str='gpt-5-nano', temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=True, stops: List[str]=None, instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.instructions = instruct
		self.stops = stops
		self.model = model
		self.tool_choice = 'auto'
		self.response_format = 'auto'
		self.tools = [ ]
		self.reasoning = [ ]
		self.prompt = None
		self.response = None
		self.file = None
		self.file_url = None
		self.image_url = None
		self.input = [ ]
		self.output_text = None
		self.include = [ ]
		self.max_search_results = None
		self.purpose = None
		self.file_ids = [ ]
		self.include = [ ]
		self.tools = [ ]
		self.vector_stores = \
		{
			'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
			'Appropriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
		}
		self.files = \
		{
			'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
			'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
			'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
			'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [
				 'gpt-5',
				 'gpt-5.2',
				 'gpt-5-mini',
				 'gpt-5-nano',
				 'gpt-5-turbo',
		         'gpt-4.1',
		         'gpt-4.1-mini',
		         'gpt-4.1-nano',
		         'gpt-4o',
		         'gpt-4o-mini' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'web_search_call.results',
		         'web_search_call.action.sources',
		         'message.input_image.image_url',
		         'computer_call_output.output.image_url',
		         'code_interpreter_call.outputs',
		         'reasoning.encrypted_content',
		         'message.output_text.logprobs' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'web_search',
		         'image_generation',
		         'file_search',
		         'code_interpreter',
		         'computer_use_preview' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'assistants',
		         'batch',
		         'fine-tune',
		         'vision',
		         'user_data',
		         'evals' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of reasoning effort options

		'''
		return [ 'low',
		         'medium',
		         'high',
		         'none',
		         'minimal',
		         'xhigh' ]
	
	def generate_text( self, prompt: str, model: str='gpt-5-nano', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=True, stream: bool=True, instruct: str=None,
			reasoning: str='low', include: str=None  ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Generates a chat completion given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			self.model = model
			self.prompt = prompt
			self.temperature = temperature
			self.reasoning = reasoning
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_completion_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instructions = instruct
			self.reasoning= {'effort': self.reasoning }
			self.input = self.prompt
			self.include = include
			self.client = OpenAI( api_key=self.api_key )
			if self.model.startswith( 'gpt-5' ):
				self.response = self.client.responses.create( model=self.model, input=self.input,
					reasoning=self.reasoning )
			else:
				self.response = self.client.responses.create( model=self.model, input=self.input,
					max_output_tokens=self.max_completion_tokens, temperature=self.temperature,
					top_p=self.top_percent )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			raise  exception
	
	def generate_image( self, prompt: str, number: int=1, model: str='dall-e-3',
			size: str='1024x1024', quality: str='standard', fmt: str='.png'  ) -> str | None:
		'''
	
	        Purpose
	        _______
	        Generates an image given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str

	
	        Returns
	        -------
	        str | None

        '''
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.number = number
			self.model = model
			self.size = size
			self.quality = quality
			self.response_format = fmt
			self.response = self.client.images.generate( model=self.model, prompt=self.prompt,
				size=self.size, quality=self.quality,
				response_format=self.response_format, n=self.number )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('generate_image( self, prompt: str ) -> str | None')
			raise exception
	
	def analyze_image( self, prompt: str, url: str ) -> str | None:
		"""

	        Purpose
	        _______
	        Analyze an image with a text instruction.
	
	        Parameters
	        ----------
	        prompt: str
	        url: str
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'url', url )
			self.prompt = prompt
			self.image_url = url
			self.input = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'input_text',
					'text': self.prompt
				},
				{
					'type': 'input_image',
					'image_url': self.image_url
				}, ],
			} ]
			
			self.response = self.client.responses.create( model=self.model, input=self.input )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def edit_image( self, prompt: str, src_url: str, dest_path: str ) -> str | None:
		'''
			
			Purpose:
			--------
			
			
			Parameters:
			---------
			prompt: str - instructions guiding the LLM
			src_url: str - The path to the source image
			dest_path: str - name of the edited image
	
			Returns:
			----------
			
			
		'''
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'src_url', src_url )
			throw_if( 'dest_path', dest_path )
			self.prompt = prompt
			_source = src_url
			_url = dest_path
			
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
			
	def summarize_document( self, prompt: str, pdf_path: str) -> str | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'pdf_path', pdf_path )
			self.prompt = prompt
			self.file_path = pdf_path
			self.file = self.client.files.create( file=open( file=self.file_path, mode='rb' ),
				purpose='user_data' )
			self.messages = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'file',
					'file':
					{
						'file_id': self.file.id,
					},
				},
				{
					'type': 'text',
					'text': self.prompt,
				},],
			}]
			
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_web( self, prompt: str, model: str='gpt-4.1-nano-2025-04-14',
			recency: int=30, max_results: int=100, ) -> str | None:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a prompt,
	
	        Parameters
	        ----------
	        prompt: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.search_recency = recency
			self.max_search_results = max_results
			self.web_options = { 'search_recency_days': self.search_recency,
			                     'max_search_results': self.max_search_results }
			self.messages = [
			{
				'role': 'user',
				'content': self.prompt,
			}]
			
			self.response = self.client.responses.create( model=self.model,
				web_search_options=self.web_options, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('search_web( self, prompt: str, model: str, '
			                    'recency: int=30, max_results: int=8 ) -> str | None')
			error = ErrorDialog( exception )
			error.show( )

	def upload_file( self, filepath: str, purpose: str='user_data' ) -> str | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'filepath', filepath )
			self.filepath = filepath
			self.purpose = purpose
			self.client = OpenAI( api_key=self.api_key )
			self.file = self.client.files.create( file=open( file=filepath, mode='rb' ), 
				purpose=self.purpose )
			return self.file.id
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'upload_file( self, filepath: str, purpose: str=user_data ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_file( self, id: str ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.retrieve( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_files( self, purpose: str = "user_data" ):
		self.purpose = purpose
		self.client = OpenAI( api_key=self.api_key )
		page = self.client.files.list( )  # no purpose arg
		files = getattr( page, "data", None ) or page
		out = [ ]
		for f in files:
			if purpose and getattr( f, "purpose", None ) != purpose:
				continue
			out.append( {
					"id": str( getattr( f, "id", "" ) ),
					"filename": str( getattr( f, "filename", "" ) ),
					"purpose": str( getattr( f, "purpose", "" ) ),
			} )
		return out
	
	def retrieve_content( self, id: str ) -> str | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.content( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_file( self, id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.files.delete( file_id=id )
			return bool( _deleted )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'delete_file( self, id: str ) -> FileDeleted '
			error = ErrorDialog( exception )
			error.show( )
	
	def create_store( self, store_name: str ) -> VectorStore | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_name', store_name )
			self.client = OpenAI( api_key=self.api_key )
			_store = self.client.vector_stores.create( name=store_name )
			return _store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'create_store( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_store( self, id: str ) -> VectorStore | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.retrieve( vector_store_id=id )
			return vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_store( self, purpose: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_store( self, id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.vector_stores.delete( vector_store_id=id )
			return bool( _deleted )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_store( self, purpose: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
		
	def __dir__( self ) -> List[ str ] | None:
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'system_instructions',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'completion',
		         'file',
		         'path',
		         'messages',
		         'image_url',
		         'response_format',
		         'tools',
		         'vector_store_ids',
		         'name',
		         'id',
		         'description',
		         'get_format_options',
		         'get_model_options',
		         'reasoning_effort',
		         'purpose_options',
		         'input_text',
		         'metadata',
		         'get_files',
		         'get_data',
		         'dump',
		         'translate',
		         'transcribe',
		         'generate_text',
		         'generate_image',
		         'analyze_image',
		         'edit_image',
		         'summarize_document',
		         'search_web',
		         'search_files',
		         'retrieve_file',
		         'retrieve_files',
		         'retrieve_content',
		         'delete_file',
		         'upload_file', ]

class Files( GPT ):
	'''
		
		Purpose:
		--------
		
		Attributes:
		----------
		
	'''
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	name: Optional[ str ]
	response_format: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	purpose: Optional[ str ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	file_ids: Optional[ List[ str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = None
		self.content = None
		self.prompt = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.input = None
		self.purpose = None
		self.documents = \
		{
				'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
				'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
				'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
				'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
		}
	
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
		return [ 'assistants',
		         'batch',
		         'fine-tune',
		         'vision',
		         'user_data',
		         'evals' ]
	
	def upload( self, filepath: str, purpose: str='user_data' ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'filepath', filepath )
			self.filepath = filepath
			self.purpose = purpose
			self.client = OpenAI( api_key=self.api_key )
			self.file = self.client.files.create( file=open( file=filepath, mode='rb' ),
				purpose=self.purpose )
			return self.file.id
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'upload_file( self, filepath: str, purpose: str=user_data ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve( self, id: str ) -> List[ str ] | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.retrieve( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def summarize( self, prompt: str, pdf_path: str ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'pdf_path', pdf_path )
			self.prompt = prompt
			self.file_path = pdf_path
			self.file = self.client.files.create( file=open( file=self.file_path, mode='rb' ),
				purpose='user_data' )
			self.messages = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'file',
											'file':
												{
														'file_id': self.file.id,
												},
									},
									{
											'type': 'text',
											'text': self.prompt,
									}, ],
					} ]
			
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def search( self, prompt: str, store_id: str, model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def survey( self, prompt: str, model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			self.prompt = prompt
			self.model = model
			self.client = OpenAI( api_key=self.api_key )
			self.vector_store_ids = list( self.vector_stores.values( ) )
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def list( self, purpose: str="user_data" ):
		self.purpose = purpose
		self.client = OpenAI( api_key=self.api_key )
		page = self.client.files.list( )  # no purpose arg
		files = getattr( page, "data", None ) or page
		out = [ ]
		for f in files:
			if purpose and getattr( f, "purpose", None ) != purpose:
				continue
			out.append( {
					"id": str( getattr( f, "id", "" ) ),
					"filename": str( getattr( f, "filename", "" ) ),
					"purpose": str( getattr( f, "purpose", "" ) ),
			} )
		return out
	
	def extract( self, id: str ) -> str | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.content( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.files.delete( file_id=id )
			return bool( _deleted )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'delete_file( self, id: str ) -> FileDeleted '
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response_format',
		         'name',
		         'purpose',
		         'content',
		         'file_id',
		         'documents',
		         'retrieve',
		         'list',
		         'extract',
		         'delete',
		         'upload', ]

class Embeddings( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for creating vectors using OpenAI's embedding models
	
	    Parameters
	    ------------
	    None
	
	    Attributes
	    -----------
	    api_key
	    client
	    model
	    embedding
	    response
	
	    Methods
	    ------------
	    create( self, text: str ) -> get_list[ float ]


    """
	response: Optional[ CreateEmbeddingResponse ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.encoding_format = None
		self.model = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		'''
		
			Returns:
			--------
			List[ str ] of embedding models

		'''
		return [ 'text-embedding-3-small',
		         'text-embedding-3-large' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float',
		         'base64' ]
	
	def create( self, text: str, model: str='text-embedding-3-small', format: str='float' ) -> List[ float ]:
		"""
	
	        Purpose
	        _______
	        Creates an embedding ginve a text
	
	
	        Parameters
	        ----------
	        text: str
	
	
	        Returns
	        -------
	        get_list[ float

        """
		try:
			throw_if( 'text', text )
			self.input = text
			self.model = model
			self.encoding_format = format
			self.response = self.client.embeddings.create( input=self.input, model=self.model,
				encoding_format=self.encoding_format )
			self.embedding = self.response.data[ 0 ].embedding
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def count_tokens( self, text: str, coding: str ) -> int:
		'''

	        Purpose:
	        -------
	        Returns the num of words in a documents path.
	
	        Parameters:
	        -----------
	        text: str - The string that is tokenized
	        coding: str - The encoding to use for tokenizing
	
	        Returns:
	        --------
	        int - The number of words

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'coding', coding )
			_encoding = tiktoken.get_encoding( coding )
			_tokens = len( _encoding.encode( text ) )
			return _tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		        Purpose:
		        --------
                Method returns a list of strings representing members

		        Parameters:
		        ----------
                self

		        Returns:
		        ---------
                List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'create',
		         'api_key',
		         'client',
		         'model',
		         'count_tokens',
		         'path',
		         'model_options', ]

class TTS( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (TTS)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants
	
	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, prompt: str, path: str )

    """
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	language: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	response: Optional[ openai.types.responses.Response ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=True, stream: bool=True, instruct: str=None ):
		'''

	        Purpose:
	        --------
	        Constructor to  create_small_embedding TTS objects

        '''
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = 'gpt-4o-mini-tts'
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_path = None
		self.response = None
		self.response_format = None
		self.speed = None
		self.voice = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Methods that returns a list of tts model names

        '''
		return [ 'gpt-4o-mini-tts',
		         'tts-1',
		         'tts-1-hd' ]

	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'verse',
		         'ballad',
		         'aria',
		         'sol',
		         'luna',
		         'nova',
		         'sage', ]
		
	@property
	def format_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3',
		         'wav',
		         'ogg',
		         'flac']
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of floats
	        representing different audio speeds

        '''
		return [ 0.25, 1.0, 4.0 ]
	
	def create_audio( self, text: str, filepath: str, format: str='mp3',
			speed: float=1.0, voice: str='aria' ) -> str:
		"""
	
	        Purpose
	        _______
	        Generates audio given a text prompt less than
	        4096 characters and a path to audio file
	
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			throw_if( 'filepath', filepath )
			self.input_text = text
			self.speed = speed
			self.response_format = format
			self.voice = voice
			out_path = Path( filepath )
			if not out_path.parent.exists( ):
				out_path.parent.mkdir( parents=True, exist_ok=True )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			with self.client.audio.speech.with_streaming_response.create( model=self.model, speed=self.speed,
					voice=self.voice, response_format=self.response_format, input=self.input_text ) as resp:
				resp.stream_to_file( str( out_path ) )
			return str( out_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_audio( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )

	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'system_instructions',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'completion',
		         'file',
		         'path',
		         'messages',
		         'image_url',
		         'response_format',
		         'tools',
		         'vector_store_ids',
		         'name',
		         'id',
		         'description',
		         'generate_text',
		         'get_format_options',
		         'get_model_options',
		         'reasoning_effort',
		         'get_effort_options',
		         'get_speed_options',
		         'input_text',
		         'metadata',
		         'get_files',
		         'get_data',
		         'dump', ]

class Transcription( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.audio_file, self.transcript
	
	
	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, path: str  ) -> str


    """
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	language: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str='gpt-4o-mini-transcribe', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=True, language: str='en', instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=self.api_key )
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.language = language
		self.instructions = instruct
		self.model = model
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
	
	@property
	def model_options( self ) -> str:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-4o-mini-transcribe'  ]
		
	@property
	def format_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3',
		         'wav',
		         'aac',
		         'flac',
		         'opus',
		         'pcm' ]
	
	@property
	def output_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			-------
			List[ str ] output  format options
			
		'''
		return [ 'json',
		         'text',
		         'srt',
		         'verbose_json',
		         'vtt',
		         'diarized_json' ]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'en',
		         'sp',
		         'fr',
		         'jp',
		         'de',
		         'it',
		         'zh', 'pt', 'ru', 'uk', 'el', 'he', 'ar', 'hi', 'ko', 'vi', 'th' ]
	
	def transcribe( self, path: str, model: str='gpt-4o-mini-transcribe', language: str='en' ) -> str:
		"""
		
			Purpose:
			----------
            Transcribe audio with Whisper.
        
        """
		try:
			throw_if( 'path', path )
			self.model = model
			self.language = language
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			with open( path, 'rb' ) as self.audio_file:
				resp = self.client.audio.transcriptions.create( model=self.model,
					file=self.audio_file, language=self.language )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'boo'
			ex.cause = 'Transcription'
			ex.method = 'transcribe(self, path)'
			error = ErrorDialog( ex )
			error.show( )

	def __dir__( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_file',
		         'messages',
		         'response_format',
		         'api_key',
		         'client',
		         'input_text',
		         'transcript', ]

class Translation( GPT ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants
	
	    Methods
	    ------------
	    create_small_embedding( self, prompt: str, path: str )

    """
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str='gpt-5-mini', temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True, instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_file = None
		self.response = None
		self.voice = None
	
	@property
	def model_options( self ) -> str:
		'''
	
	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-5',
		         'gpt-5-mini',
		         'gpt-5-turbo',
		         'gpt-4o',
		         'gpt-4.1',
		         'gpt-4.1-mini',
		         'gpt-4.1-turbo',
		         'gpt-4o',
		         'gpt-4o-mini',]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of translatable languages

        '''
		return [ 'english'
				'spanish'
				'french'
				'german'
				'italian'
				'portuguese'
				'russian'
				'ukrainian'
				'greek'
				'hebrew'
				'arabic'
				'hindi'
				'chinese'
				'japanese'
				'korean'
				'vietnamese'
				'thai' ]
	
	@property
	def voice_options( self ):
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'ash',
		         'ballad',
		         'coral',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'sage',
		         'shiver', ]
	
	
	def translate( self, prompt: str, path: str, model: str = 'gpt-5-mini',
			temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 10000,
			store: bool = True, stream: bool = True, instruct: str = None ) -> str | None:
		"""
		
            Translate non-English speech to English with Whisper.
        
        """
		try:
			throw_if( 'path', path )
			throw_if( 'prompt', prompt )
			self.model = model
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_completion_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instruct = instruct
			with open( path, 'rb' ) as audio_file:
				self.response = self.client.audio.translations.create( model=self.model,
					file=audio_file )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'boo'
			ex.cause = 'Translation'
			ex.method = 'translate(self, path)'
			error = ErrorDialog( ex )
			error.show( )

	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_path',
		         'path',
		         'messages',
		         'response_format',
		         'tools',
		         'api_key',
		         'client',
		         'model',
		         'translate',
		         'model_options', ]

class Images( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for generating images OpenAI's Images API and dall-e-2
	
	
	    Parameters
	    ------------
	    n: int=1
	    temperature: float=0.8
	    top_p: float=0.9
	    frequency: float=0.0
	    presence: float=0.0
	    max_tokens: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.client, self.small_model,  self.embedding,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.prompt, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.input_text, self.image_url
	
		Properties:
		----------
	    detail_options( self ) -> list[ str ]
	    format_options( self ) -> list[ str ]
	    size_options( self ) -> list[ str ]
	    model_options( self ) -> str
	    
	    Methods
	    ------------
	    generate( self, path: str ) -> str
	    analyze( self, path: str, text: str ) -> str

    """
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	file_url: Optional[ str ]
	style: Optional[ str ]
	response_format: Optional[ str ]
	background: Optional[ str ]
	
	def __init__( self, temperture: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=False, stream: bool=False,
			instruct: str=None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=self.api_key )
		self.temperature = temperture
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instruct = instruct
		self.tools = [ ]
		self.tool_choice = 'auto'
		self.input = [ ]
		self.background = None
		self.input_text = None
		self.file_path = None
		self.image_url = None
		self.quality = None
		self.detail = None
		self.model = None
		self.size = None
		self.style = None
		self.response_format = None
	
	@property
	def style_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Methods that returns a list of style options for dall-e-3

        '''
		return [ 'vivid',
		         'natural', ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-image-1',
		         'gpt-image-1-mini',
		         'gpt-image-1.5',
		         'gpt-5',
		         'gpt-5.2',
		         'gpt-5-mini',
		         'gpt-5-nano',
		         'gpt-5-turbo',
		         'gpt-4.1',
		         'gpt-4.1-mini',
		         'gpt-4o',
		         'gpt-4o-mini' ]
		
	@property
	def size_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Method that returns a  list of sizes

	        - For gpt-image-1, the size must be one of '1024x1024', '1536x1024' (landscape),
	        '1024x1536' (portrait), or 'auto' (default value).

	        - For dall-e-2, the size must be one of '256x256', '512x512', or '1024x1024'

	        - For dall-e-3, the sie must be one of '1024x1024', '1792x1024', or '1024x1792'

        '''
		return [ '256x256',
		         '512x512',
		         '1024x1024' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'png',
		         'jpeg',
		         'jpg',
		         'webp',
		         'gif' ]
	
	@property
	def output_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'png',
		         'jpeg',
		         'webp' ]
	
	@property
	def background_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'transparent',
		         'opaque',
		         'auto' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of quality options

        '''
		return [ 'standard', 'hd' ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of detail options

        '''
		return [ 'auto',
		         'low',
		         'high' ]
	
	def generate( self, prompt: str, model: str, quality: str, size: str,
			style: str='natural', format: str='png', instruct: str=None ) -> str:
		"""

                Purpose
                _______
                Generate an image from a text prompt.


                Parameters
                ----------
                prompt: str


                Returns
                -------
                Image object

        """
		try:
			throw_if( 'text', prompt )
			throw_if( 'model', model )
			throw_if( 'quality', quality )
			throw_if( 'size', size )
			self.input_text = prompt
			self.model = model
			self.quality = quality
			self.size = size
			self.style = style
			self.response_format = format
			self.tools.append({'type':'image_generation'})
			self.instruct = instruct
			self.response = self.client.responses.create( model=self.model, input=self.input_text,
				size=self.size, style=self.style, response_format=self.response_format,
				tools=self.tools, quality=self.quality  )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'generate( self, path: str ) -> str'
			raise exception
	
	def analyze( self, text: str, path: str, model: str='gpt-4o-mini', ) -> str:
		'''
	
	        Purpose:
	        ________
	
	        Method providing image analysis functionality given a prompt and path
	
	        Parameters:
	        ----------
	        input: str
	        path: str
	
	        Returns:
	        --------
	        str | None

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			self.input_text = text
			self.model = model
			self.file_path = path
			self.input = [
			{
				'role': 'user',
				'content': [
					{ 'type': 'input_text', 'text': self.input_text },
					{ 'type': 'input_image', 'image_url': self.file_path },
				],
			}]
			
			self.response = self.client.responses.create( model=self.model, input=self.input,
				max_output_tokens=self.max_completion_tokens, temperature=self.temperature,
				tool_choice=self.tool_choice, stream=self.stream, store=self.store )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'analyze( self, path: str, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def edit( self, prompt: str, path: str, size: str='1024x1024' ) -> str:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a path prompt,
	
	        Parameters
	        ----------
	        prompt: str
	        url: str
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'input', prompt )
			throw_if( 'path', path )
			self.input_text = prompt
			self.file_path = path
			self.response = self.client.images.edit( model=self.model,
				image=open( self.file_path, 'rb' ), prompt=self.input_text, n=self.number,
				size=self.size, )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'edit( self, text: str, path: str, size: str=1024x1024 ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ # Attributes
				 'number',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'api_key',
		         'client',
		         'path',
		         'input_text',
		         'image_url',
		         'size',
		         'quality',
		         'detail',
		         'model',
		         # Properties
				 'style_options',
		         'model_options',
		         'detail_options',
		         'format_options',
		         'size_options',
		         # Methods
		         'generate',
		         'analyze',
		         'edit', ]

class VectorStores( GPT ):
	'''
		
		Purpose:
		--------
		
		Attributes:
		----------
		
	'''
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	name: Optional[ str ]
	store_ids: Optional[ List[ str ] ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	collections: Optional[ Dict[ str, Any ] ]
	
	def __init__( self  ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = None
		self.name = None
		self.content = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.collections = \
		{
			'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
			'Appropriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
		}
		self.documents = \
		{
			'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
			'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
			'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
			'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
		}

	def create( self, store_name: str ) -> Any | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_name', store_name )
			self.client = OpenAI( api_key=self.api_key )
			_store = self.client.vector_stores.create( name=store_name )
			return _store
		except Exception as e:
			raise
	
	def list( self ) -> Any | None:
		self.client = OpenAI( api_key=self.api_key )
		_stores = self.client.vector_stores.list( )
		return _stores
	
	def retrieve( self, id: str ) -> Any | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.retrieve( vector_store_id=id )
			return vector_store
		except Exception as e:
			raise
	
	def search( self, prompt: str, store_id: str, model: str='gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			raise
	
	def survey( self, prompt: str, store_id: List[ str ]=None,
			model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			raise
	
	def update( self, store_id: str, filename: str ):
		try:
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.update( vector_store_id=store_id, name=filename )
			return vector_store
		except Exception as e:
			return None
	
	def delete( self, store_id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_id', store_id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.vector_stores.delete( vector_store_id=id )
			return bool( _deleted )
		except Exception as e:
			return None
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response_format',
		         'name',
		         'content',
		         'file_id',
		         'collections',
		         'retrieve',
		         'list',
		         'extract',
		         'delete',
		         'update', ]
		