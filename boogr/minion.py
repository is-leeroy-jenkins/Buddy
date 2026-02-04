'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                minion.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="minion.py" company="Terry D. Eppler">

         Buddy is a df analysis tool integrating various Generative AI, Text-Processing, and
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
    minion.py
  </summary>
  ******************************************************************************************
  '''
import subprocess as sp
from .static import Client

class App( ):
    '''

        Constructor:
        -------------
        App( client: enum )


        Purpose:
        ---------
        Class defines object providing factory methods run( ) and run( args ) that run
        processes based on 'Client' enumeration path args

    '''

    def __init__( self, client: Client ):
        self.app = client
        self.sqlite = r'target_values\sqlite\gui\SQLiteDatabaseBrowserPortable.exe'
        self.access = r'C:\Program Files\Microsoft Office\root\Office16\MSACCESS.EXE'
        self.excel = r'C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE'
        self.edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        self.chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
        self.control_panel = r'C:\Windows\System32\control.exe'
        self.calculator = r'C:\Windows\System32\calc.exe'
        self.outlook = r'C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE'
        self.pyscripter = r'target_values\python\PyScripter\PyScripter.exe'
        self.task_manager = r'C:\Windows\System32\Taskmgr.exe'
        self.storage = r'C:\Users\terry\AppData\Local\Microsoft\OneDrive\OneDrive.exe'
        self.word = r'C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE'

    def __dir__( self ) -> list[ str ]:
	    return [ 'sqlite', 'access', 'excel', 'chrome',
		         'edge', 'control', 'calculator', 'task_manager',
		         'outlook', 'pyscripter', 'storage', 'word',
		         'run', 'run_args' ]

    def run( self ):
        '''

            Purpose:
            ---------

            Parameters:
            ----------

            Returns:
            -------

        '''
        try:
            if self.app == Client.SQLite:
                sp.Popen( self.sqlite )
            elif self.app == Client.Access:
                sp.Popen( self.access )
            elif self.app == Client.Excel:
                sp.Popen( self.excel )
            elif self.app == Client.Edge:
                sp.Popen( self.edge )
            elif self.app == Client.Chrome:
                sp.Popen( self.chrome )
            elif self.app == Client.ControlPanel:
                sp.Popen( self.control_panel )
            elif self.app == Client.Calculator:
                sp.Popen( self.calculator )
            elif self.app == Client.Outlook:
                sp.Popen( self.outlook )
            elif self.app == Client.Pyscripter:
                sp.Popen( self.pyscripter )
            elif self.app == Client.TaskManager:
                sp.Popen( self.task_manager )
            elif self.app == Client.Storage:
                sp.Popen( self.storage )
            elif self.app == Client.Word:
                sp.Popen( self.word )
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Minion'
            _exc.cause = 'App'
            _exc.method = 'run( self )'
            _err = ErrorDialog( _exc )
            _err.show( )

    def run_args( self, args: str ):
        '''

            Purpose:
            -------

            Parameters:
            -----------

            Returns:
            ---------

        '''
        try:
            if args is not None and self.app == Client.SQLite:
                if os.path.isfile( args ):
                    sp.Popen( [ self.sqlite, args ] )
            elif args is not None and self.app == Client.Access:
                if os.path.isfile( args ):
                    sp.Popen( [ self.access, args ] )
            elif args is not None and self.app == Client.Excel:
                if os.path.isfile( args ):
                    sp.Popen( [ self.excel, args ] )
            elif args is not None and self.app == Client.Edge:
                    sp.Popen( [ self.edge, args ] )
            elif args is not None and self.app == Client.Chrome:
                    sp.Popen( [ self.chrome, args ] )
            elif args is not None and self.app == Client.Outlook:
                    sp.Popen( [ self.outlook, args ] )
            elif args is not None and self.app == Client.Pyscripter:
                    sp.Popen( [ self.pyscripter, args ] )
            elif args is not None and self.app == Client.Word:
                    sp.Popen( [ self.__word, args ] )
            elif args is not None and self.app == Client.TaskManager:
                    sp.Popen( [ self.task_manager, args ] )
            elif args is not None and self.app == Client.ControlPanel:
                    sp.Popen( [ self.control_panel, args ] )
            elif args is not None and self.app == Client.Storage:
                    sp.Popen( [ self.storage, args ] )
        except Exception as e:
            _exc = Error( e )
            _exc.module = 'Minion'
            _exc.cause = 'App'
            _exc.method = 'run_args( self, args )'
            _err = ErrorDialog( _exc )
            _err.show( )