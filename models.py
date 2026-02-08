'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                models.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="models.py" company="Terry D. Eppler">

	     models.py
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
    models.py
  </summary>
  ******************************************************************************************
'''
from typing import Any, Dict, List, Optional

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Prompt( ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Boo when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		instructions: Optional[ str ]
			The primary instruction block (typically the system message content).

		context: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		output_indicator: Optional[ str ]
			A short indicator describing the desired output style/format (e.g., "json", "table").

		input_data: Optional[ str ]
			Optional data payload embedded into the prompt (small inputs, examples, etc.).

		id: Optional[ str ]
			Optional identifier for tracking prompts (e.g., GUID, hash, or friendly name).

		version: Optional[ str ]
			Optional version string for prompt management and experimentation.

		format: Optional[ str ]
			Optional format label describing the prompt template type (e.g., "chat", "completion").

		variables: Optional[ List[ str ] ]
			Optional list of placeholder variables referenced by the prompt template.

		question: Optional[ str ]
			Optional question or user query associated with the prompt.

	'''
	instructions: Optional[ str ]
	content: Optional[ str ]
	variables: Optional[ List[ str ] ]
	data: Optional[ Dict[ str, Any ] ]
	id: Optional[ str ]
	version: Optional[ str ]
	
	def __init__( self, instructions: str=None, content: str=None, variables: List[ str ]=None,
			data: Dict[ str, Any ]=None, id: int=None, version: str=None ):
		self.instructions = instructions
		self.content = content
		self.variables = variables
		self.data = data
		self.id = id
		self.version = version
		
	def __dir__( self ) -> List[ str ]:
		'''
		
			Purpose:
			--------
			Provides a list of public class members
			
			Returns:
			--------
			List[ str ]
			
		
		'''
		return [ 'instructions', 'content', 'variables', 'data', 'id', 'version' ]

	def __str__( self ):
		'''
		
			Purpose:
			---------
			Returns a string representation of the model
			
		'''
		if self.content is not None:
			return self.content
		elif self.instructions is not None:
			return self.instructions
		else:
			return None
			
	
class Message( ):
	'''

		Purpose:
		--------
		Represents a chat message-like object used by Boo to normalize conversational state.
		This is intentionally general to support both “input messages” and “output messages”.

		Attributes:
		----------
		content: str
			Message content payload. Boo treats this as required for operational messages.

		role: str
			Message role (e.g., "system", "user", "assistant", "tool").

		type: Optional[ str ]
			Optional discriminator if an upstream system emits typed message objects.

		instructions: Optional[ str ]
			Optional per-message instruction string (used in some orchestration patterns).

		data: Optional[ Dict ]
			Optional message metadata or additional structured payload.

	'''
	content: Optional[ str ]
	role: Optional[ str ]
	type: Optional[ str ]
	
	def __init__( self, role: str=None, content: str=None,  type: str=None  ):
		self.role = role
		self.content = content
		self.type = type
	
	def __dir__( self ) -> List[ str ]:
		'''
			
			Purpose:
			--------
			Provides a list of public class members.

			Returns:
			--------
			List[ str ]
		
		'''
		return[ 'content', 'role', 'type' ]

	def __str__( self ):
		'''
		
			Purpose:
			--------
			Provides a string representation of this message.
			
		'''
		return self.content
		
class Payload( ):
	'''

		Purpose:
		--------
		Represents a general-purpose request payload. This model intentionally
		supports both request parameters (as echoed back) and response payload f
		ields, while remaining permissive to vendor
		evolution via `extra='ignore'`.

		Attributes:
		----------
		response_format: Optional[ ResponseFormat ]
		The response format settings used or returned by the upstream system.

		temperature: Optional[ float ]
		Optional. Controls the randomness of the output.
		Values can range from [0.0, 2.0].

		top_p: Optional[ float ]
		Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.

		max_output_tokens: Optional[ int ]
		Max output tokens cap.

		store: Optional[ bool ]
		Whether the upstream should store the response (provider-specific).

		stream: Optional[ bool ]
		Whether streaming was requested.
		
		top_k: Optional[ int ]
		Optional. The maximum number of tokens to consider when sampling.
		Gemini models use Top-p (nucleus) sampling or a combination of Top-k and nucleus sampling.
		Top-k sampling considers the set of topK most probable tokens. Models running with nucleus
		sampling don't allow topK setting.
		
		presence_penalty: Optional[ float ]
		Optional. Presence penalty applied to the next token's logprobs if the token has already
		been seen in the response. This penalty is binary on/off and not dependant on the
		number of times the token is used (after the first).
		
		frequency_penalty: Optional[float ]
		Optional. Frequency penalty applied to the next token's logprobs, multiplied by the number
		of times each token has been seen in the respponse so far. A positive penalty will
		discourage the use of tokens that have already been used, proportional to the number of
		times the token has been used: The more a token is used, the more difficult it is for
		the model to use that token again increasing the vocabulary of responses.
		
		logprobs: Optional[ int ]
		Optional. Only valid if responseLogprobs=True. This sets the number of top logprobs to
		return at each decoding step in the Candidate.logprobs_result.
		The number must be in the range of [0, 20].
		
		stops: Optional[ List[str] ]
		Optional. The set of character sequences (up to 5) that will stop output generation.
		If specified, the API will stop at the first appearance of a stop_sequence.
		The stop sequence will not be included as part of the response.
		
	'''
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	logprobs: Optional[ int ]
	max_tokens: Optional[ int ]
	presence_penalty: Optional[ float ]
	fequency_penalty: Optional[ float ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	stop_sequence: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	
	def __init__( self, temperature: float=None, top_p: float=None, top_k: int=None,
			logprobs: int=None, presense: float=None, store: bool=None,
			stream: bool=None, stops: List[ str ]=None, format: str=None ):
		self.temperature = temperature
		self.top_p = top_p
		self.top_k = top_k
		self.logprobs = logprobs
		self.presense = presense
		self.store = store
		self.stream = stream
		self.stop_sequence = stops
		self.response_format = format
	
	def __dir__( self ) -> List[ str ]:
		'''
			
			Purpose:
			--------
			Provides a list of public members of the class.
			
			Returns:
			--------
			List[ str ]
		
		'''
		return [ 'temperature', 'top_p', 'top_k', 'logprobs', 'presense',
		         'store', 'stream', 'stop_sequence', 'response_format' ]