'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                init.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="init.py" company="Terry D. Eppler">

	     init.py is part of a data analysis tool integrating GenAI, Text Processing,
	     and Machine-Learning algorithms for federal analysts.
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
    init.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations

import traceback
from sys import exc_info
from typing import List, Tuple, Optional
import re
from datetime import datetime, timedelta
from pathlib import Path
import config as cfg
import sqlite3

def get_config_bool( name: str, default: bool = False ) -> bool:
	"""Read a Boolean value from the configuration module.

	Purpose:
		Safely read a Boolean configuration value without requiring every deployment of Fiddy to
		define the newest configuration switches. Missing values return the supplied default.

	Args:
		name (str): Configuration attribute name.
		default (bool): Default value used when the attribute is not available.

	Returns:
		bool: Configuration Boolean value or the supplied default.
	"""
	try:
		value = getattr( cfg, name, default )
		return bool( value )
	except Exception:
		return default

def get_config_int( name: str, default: int ) -> int:
	"""Read an integer value from the configuration module.

	Purpose:
		Safely read an integer configuration value without requiring every deployment of Fiddy to
		define the newest configuration switches. Missing or invalid values return the supplied
		default.

	Args:
		name (str): Configuration attribute name.
		default (int): Default value used when the attribute is not available or invalid.

	Returns:
		int: Configuration integer value or the supplied default.
	"""
	try:
		value = getattr( cfg, name, default )
		return int( value )
	except Exception:
		return default

def sanitize_text( value: object, max_length: int = 1000 ) -> str:
	"""Return a reviewer-safe and log-safe text representation.

	Purpose:
		Remove or mask sensitive values before any exception metadata is persisted. This helper
		masks email addresses, Windows paths, POSIX paths, temporary-file paths, common uploaded
		file references, long quoted payloads, long comma-delimited manifest-like rows, and long
		free-form OCR-like text. Raw text logging can only be enabled explicitly through
		``cfg.ENABLE_RAW_TEXT_LOGGING``.

	Args:
		value (object): Source value to sanitize.
		max_length (int): Maximum returned string length after masking.

	Returns:
		str: Sanitized text value suitable for local diagnostic logging.
	"""
	try:
		if value is None:
			return ''
		
		text = str( value )
		text = text.replace( '\x00', '' )
		text = re.sub( r'[\r\n\t]+', ' ', text )
		text = re.sub( r'\s{2,}', ' ', text ).strip( )
		
		if not get_config_bool( 'ENABLE_FILE_PATH_LOGGING', False ):
			text = re.sub( r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*', '[PATH]',
				text )
			text = re.sub( r'/(?:tmp|mnt|var|home|Users|user|workspace|app|data)(?:/[^\s,;:)]*)+',
				'[PATH]', text )
			text = re.sub( r'\\(?:tmp|temp|users|appdata|programdata)\\[^\s,;:)]*', '[PATH]', text,
				flags=re.IGNORECASE )
		
		text = re.sub( r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL]', text,
			flags=re.IGNORECASE )
		text = re.sub( r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
			'[GUID]', text, flags=re.IGNORECASE )
		text = re.sub(
			r'(?i)(password|secret|token|api[_-]?key|subscription[_-]?key)\s*[=:]\s*[^\s,;]+',
			r'\1=[SECRET]', text )
		
		if not get_config_bool( 'ENABLE_RAW_TEXT_LOGGING', False ):
			text = re.sub( r'"[^"]{120,}"', '"[LONG_TEXT]"', text )
			text = re.sub( r"'[^']{120,}'", "'[LONG_TEXT]'", text )
			
			if len( text ) > 300 and text.count( ',' ) >= 5:
				text = '[MANIFEST_ROW]'
			elif len( text ) > 500 and len( re.findall( r'\b[A-Za-z]{3,}\b', text ) ) > 60:
				text = '[LONG_TEXT]'
		
		return text[ :max_length ]
	except Exception:
		return '[SANITIZATION_FAILED]'

def sanitize_traceback( value: object, max_length: int = 4000 ) -> str:
	"""Return a sanitized traceback string.

	Purpose:
		Normalize and mask traceback text before it is stored in SQLite. This function delegates to
		``sanitize_text`` and uses a larger maximum length appropriate for diagnostic stack traces.

	Args:
		value (object): Traceback value to sanitize.
		max_length (int): Maximum returned string length.

	Returns:
		str: Sanitized traceback text.
	"""
	try:
		return sanitize_text( value, max_length=max_length )
	except Exception:
		return '[TRACE_SANITIZATION_FAILED]'

class Error( Exception ):
	"""Wrap a Python exception with structured and sanitized diagnostic metadata.

	The ``Error`` class extends ``Exception`` and stores the original exception together with
	context fields used by Fiddy logging and diagnostics. The wrapper captures a sanitized
	message, exception type, sanitized traceback, component or class cause, module name, method
	or function signature, optional heading, and combined information string.

	The object remains intentionally lightweight. Callers generally create an ``Error`` inside an
	``except`` block, assign stable metadata fields such as ``cause``, ``module``, and ``method``,
	and then pass the object to ``Logger.write`` for local persistence.

	Attributes:
		error (Optional[Exception]): Source exception being wrapped.
		heading (Optional[str]): Optional sanitized user-facing heading or category.
		cause (Optional[str]): Sanitized component, class, or module purpose associated with the failure.
		method (Optional[str]): Sanitized method or function signature associated with the failure.
		module (Optional[str]): Sanitized module name associated with the failure.
		type (Optional[type]): Exception type captured from ``sys.exc_info``.
		trace (Optional[str]): Sanitized formatted traceback captured at wrapper creation time.
		info (Optional[str]): Combined sanitized exception type and traceback information.
		message (Optional[str]): Sanitized string representation of the source exception.
	"""
	
	error: Optional[ Exception ]
	heading: Optional[ str ]
	cause: Optional[ str ]
	method: Optional[ str ]
	module: Optional[ str ]
	type: Optional[ type ]
	trace: Optional[ str ]
	info: Optional[ str ]
	message: Optional[ str ]
	
	def __init__( self, error: Exception, heading: str = None, cause: str = None,
			method: str = None, module: str = None ) -> None:
		"""Initialize an error wrapper from a caught exception.

		Purpose:
			Store the original exception and optional context values, initialize the base
			``Exception`` with the sanitized exception message, capture the current exception type,
			capture a sanitized formatted traceback, and build a combined sanitized information
			string suitable for database logging.

		Args:
			error (Exception): Source exception being wrapped.
			heading (str): Optional user-facing heading.
			cause (str): Optional component, class, or module purpose that caused the error.
			method (str): Optional stable method or function signature where the error occurred.
			module (str): Optional module where the error occurred.

		Returns:
			None.
		"""
		self.error = error
		self.heading = sanitize_text( heading, 120 ) if heading else None
		self.cause = sanitize_text( cause, 120 ) if cause else None
		self.method = sanitize_text( method, 180 ) if method else None
		self.module = sanitize_text( module, 120 ) if module else None
		self.type = exc_info( )[ 0 ]
		self.message = sanitize_text( str( error ) if error else '',
			get_config_int( 'MAX_LOG_MESSAGE_CHARS', 1000 ) )
		self.trace = sanitize_traceback( traceback.format_exc( ),
			get_config_int( 'MAX_LOG_TRACE_CHARS', 4000 ) )
		self.info = sanitize_text( f'{str( self.type )}: {self.trace}',
			get_config_int( 'MAX_LOG_TRACE_CHARS', 4000 ) )
		super( ).__init__( self.message )
	
	def __setattr__( self, name: str, value: object ) -> None:
		"""Sanitize public metadata fields when they are assigned after construction.

		Purpose:
			Preserve the established Fiddy usage pattern where callers create ``Error(e)`` and then
			assign ``cause``, ``module``, and ``method`` afterward. Metadata assignments are
			sanitized automatically so delayed assignment does not bypass privacy controls.

		Args:
			name (str): Attribute name being assigned.
			value (object): Attribute value being assigned.

		Returns:
			None.
		"""
		if name in ('heading', 'cause', 'module') and value is not None:
			object.__setattr__( self, name, sanitize_text( value, 120 ) )
		elif name == 'method' and value is not None:
			object.__setattr__( self, name, sanitize_text( value, 180 ) )
		elif name == 'message' and value is not None:
			object.__setattr__( self, name,
				sanitize_text( value, get_config_int( 'MAX_LOG_MESSAGE_CHARS', 1000 ) ) )
		elif name in ('trace', 'info') and value is not None:
			object.__setattr__( self, name,
				sanitize_traceback( value, get_config_int( 'MAX_LOG_TRACE_CHARS', 4000 ) ) )
		else:
			object.__setattr__( self, name, value )
	
	def __str__( self ) -> str:
		"""Return a string representation of the wrapped error.

		Purpose:
			Return the sanitized information string when available, otherwise return the sanitized
			source exception message. If neither value is available, return an empty string.

		

		Returns:
			str: Sanitized error information string.
		"""
		return self.info or self.message or ''
	
	def __dir__( self ) -> List[ str ]:
		"""Return public member names used by callers and display surfaces.

		Purpose:
			Expose the primary fields that are useful for logging, diagnostics, and reviewer-safe
			display. Implementation details and inherited exception internals are intentionally
			omitted from the returned member list.

		

		Returns:
			List[str]: Public error member names.
		"""
		return [
				'message',
				'cause',
				'error',
				'method',
				'module',
				'trace',
				'info'
		]

class Logger( ):
	"""Persist sanitized ``Error`` objects to the configured local SQLite database.

	The ``Logger`` class resolves the configured log database path and table name, creates the
	database table when needed, truncates sanitized values to fit configured limits, writes one
	row per error, and optionally purges old records. The logger is intentionally defensive:
	logger failures return conservative fallback values rather than raising secondary exceptions
	into application workflows.

	Attributes:
		path (Path): Resolved SQLite database path.
		table_name (str): Safe SQLite table name used for inserts and maintenance.
	"""
	path: Path
	table_name: str
	
	def __init__( self ) -> None:
		"""Initialize the logger from configured database and table settings.

		Purpose:
			Resolve the SQLite database path through ``get_database_path`` and the table name
			through ``get_table_name``. No database connection is opened until ``ensure_database``,
			``write``, or ``purge_old_logs`` is called.

		

		Returns:
			None.
		"""
		self.path = self.get_database_path( )
		self.table_name = self.get_table_name( )
	
	def get_database_path( self ) -> Path:
		"""Return the configured SQLite database path.

		Purpose:
			Read ``cfg.LOG_PATH`` and resolve it to an absolute ``Path``. Relative log paths are
			resolved under ``cfg.ROOT_DIR`` when available. If path resolution fails, the fallback
			path ``logging/Exceptions.db`` is resolved relative to the current working directory.

		

		Returns:
			Path: Resolved SQLite database path.
		"""
		try:
			value = getattr( cfg, 'LOG_PATH', 'logging/Exceptions.db' )
			path = Path( value )
			
			if not path.is_absolute( ):
				path = getattr( cfg, 'ROOT_DIR', Path.cwd( ) ) / path
			
			return path.resolve( )
		except Exception:
			return Path( 'logging/Exceptions.db' ).resolve( )
	
	def get_table_name( self ) -> str:
		"""Return a safe SQLite table name.

		Purpose:
			Read ``cfg.LOG_FILE`` and accept it only when it matches a conservative SQLite
			identifier pattern beginning with a letter or underscore and containing only letters,
			digits, and underscores. Unsafe or unavailable values fall back to ``Exceptions``.

		

		Returns:
			str: Safe SQLite table name.
		"""
		try:
			value = str( getattr( cfg, 'LOG_FILE', 'Exceptions' ) ).strip( )
			
			if re.fullmatch( r'[A-Za-z_][A-Za-z0-9_]*', value ):
				return value
			
			return 'Exceptions'
		except Exception:
			return 'Exceptions'
	
	def truncate( self, value: object, length: int ) -> str:
		"""Sanitize, convert, and truncate a value to a maximum length.

		Purpose:
			Standardize values before database insertion. ``None`` values become empty strings, all
			other values are sanitized and converted to text, and the text is sliced to the requested
			length so it fits the logging schema.

		Args:
			value (object): Source value to sanitize, convert, and truncate.
			length (int): Maximum string length.

		Returns:
			str: Sanitized and truncated string, or an empty string when conversion fails.
		"""
		try:
			if value is None:
				return ''
			
			text = sanitize_text( value, max_length=length )
			return text[ :length ]
		except Exception:
			return ''
	
	def ensure_database( self ) -> None:
		"""Create the log directory and configured error table when needed.

		Purpose:
			Create the parent directory for the configured SQLite database path and create the
			configured table if it does not already exist. The table stores timestamp, cause, module,
			method, message, info, and trace values, with an autoincrementing primary key.

		

		Returns:
			None.
		"""
		try:
			if not get_config_bool( 'ENABLE_EXCEPTION_LOGGING', True ):
				return None
			
			self.path.parent.mkdir( parents=True, exist_ok=True )
			sql = f'''
			CREATE TABLE IF NOT EXISTS "{self.table_name}" (
				"ID" INTEGER NOT NULL UNIQUE,
				"created_on" TEXT NOT NULL,
				"cause" TEXT,
				"module" TEXT,
				"method" TEXT,
				"message" TEXT,
				"info" TEXT,
				"trace" TEXT,
				PRIMARY KEY("ID" AUTOINCREMENT)
			)
			'''
			
			with sqlite3.connect( self.path ) as connection:
				connection.execute( sql )
				connection.commit( )
		except Exception:
			return None
	
	def write( self, error: Error ) -> int:
		"""Write one sanitized ``Error`` object to the configured SQLite error table.

		Purpose:
			Ensure the database exists, build a parameterized insert statement, sanitize and truncate
			error fields, write the row, commit the transaction, optionally purge expired log rows,
			and return the inserted row identifier. If exception logging is disabled or logging fails,
			return ``0`` without raising another exception.

		Args:
			error (Error): Error object to persist.

		Returns:
			int: Inserted row identifier, or ``0`` when the error is missing, disabled, or logging fails.
		"""
		try:
			if not error or not get_config_bool( 'ENABLE_EXCEPTION_LOGGING', True ):
				return 0
			
			self.ensure_database( )
			sql = f'''
			INSERT INTO "{self.table_name}"
			(
				"created_on",
				"cause",
				"module",
				"method",
				"message",
				"info",
				"trace"
			)
			VALUES
			(
				?,
				?,
				?,
				?,
				?,
				?,
				?
			)
			'''
			
			values = (datetime.utcnow( ).strftime( '%Y-%m-%d %H:%M:%S' ),
			          self.truncate( error.cause, 120 ),
			          self.truncate( error.module, 120 ),
			          self.truncate( error.method, 180 ),
			          self.truncate( error.message,
				          get_config_int( 'MAX_LOG_MESSAGE_CHARS', 1000 ) ),
			          self.truncate( error.info, get_config_int( 'MAX_LOG_TRACE_CHARS', 4000 ) ),
			          self.truncate( error.trace, get_config_int( 'MAX_LOG_TRACE_CHARS', 4000 ) ))
			
			with sqlite3.connect( self.path ) as connection:
				cursor = connection.execute( sql, values )
				connection.commit( )
				row_id = int( cursor.lastrowid or 0 )
			
			self.purge_old_logs( )
			return row_id
		except Exception:
			return 0
	
	def purge_old_logs( self, retention_days: int = None ) -> int:
		"""Delete log rows older than the configured retention period.

		Purpose:
			Support the prototype data-handling requirement by keeping local diagnostic logs short
			lived. The method deletes rows whose ``created_on`` timestamp is older than the configured
			retention window. If retention is set to zero or a negative value, no purge is performed.

		Args:
			retention_days (int): Optional retention window in days. When omitted, the value is read
				from ``cfg.LOG_RETENTION_DAYS`` and defaults to seven days.

		Returns:
			int: Number of deleted rows, or ``0`` when no rows are deleted or purge fails.
		"""
		try:
			self.ensure_database( )
			days = retention_days if retention_days is not None else get_config_int(
				'LOG_RETENTION_DAYS', 7 )
			
			if days <= 0:
				return 0
			
			cutoff = datetime.utcnow( ) - timedelta( days=days )
			sql = f'DELETE FROM "{self.table_name}" WHERE "created_on" < ?'
			
			with sqlite3.connect( self.path ) as connection:
				cursor = connection.execute( sql, (cutoff.strftime( '%Y-%m-%d %H:%M:%S' ),) )
				connection.commit( )
				return int( cursor.rowcount or 0 )
		except Exception:
			return 0

def log_error( error: Exception, heading: str = None, cause: str = None,
		method: str = None, module: str = None ) -> Error:
	"""Wrap and log an exception using the configured Fiddy error database.

	Purpose:
		Create an ``Error`` object and persist it in one step. The wrapped error object is returned
		so callers can continue to inspect or display structured metadata after logging.

	Args:
		error (Exception): Source exception being wrapped and logged.
		heading (str): Optional user-facing heading.
		cause (str): Optional component, class, or module purpose that caused the error.
		method (str): Optional stable method or function signature where the error occurred.
		module (str): Optional module where the error occurred.

	Returns:
		Error: Wrapped and sanitized error object.
	"""
	exception = Error( error=error, heading=heading, cause=cause,
		method=method, module=module )
	
	Logger( ).write( exception )
	return exception