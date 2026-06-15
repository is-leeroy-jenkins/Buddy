/*
 * ==========================================================================================
 *  Buddy Documentation JavaScript
 *  File: docs/assets/js/buddy.js
 *
 *  Purpose:
 *      Provides safe, progressive enhancements for the Buddy MkDocs Material site.
 *      This script does not require external libraries and is designed to fail gracefully
 *      when optional page elements are not present.
 *
 *  Features:
 *      - Scroll-to-top control
 *      - Heading anchor copy buttons
 *      - Large-table filtering
 *      - Code-block language labels
 *      - Code-block expand/collapse for long examples
 *      - External-link hardening
 *      - Search placeholder customization
 *      - Active navigation scroll memory
 *      - Page path metadata
 *      - Keyboard focus mode
 *      - Copy current page link button
 *      - API reference member badges
 *      - Table of contents progress marker
 *      - Local section reading progress
 *      - Print helper button
 *      - Mermaid-safe no-op guard
 *
 *  Compatibility:
 *      - MkDocs Material
 *      - Modern Chromium, Edge, Firefox, Safari
 *
 *  Notes:
 *      This file intentionally avoids external dependencies, network calls, analytics,
 *      cookies, and local storage values that contain user content.
 * ==========================================================================================
 */
( function()
{
	"use strict";
	const BuddyDocs = {
		config: {
			scrollTopId: "buddy-scroll-top",
			pageToolsId: "buddy-page-tools",
			progressId: "buddy-reading-progress",
			tableFilterClass: "buddy-table-filter",
			headingLinkClass: "buddy-heading-link",
			codeLabelClass: "buddy-code-label",
			codeToggleClass: "buddy-code-toggle",
			initializedAttribute: "data-buddy-enhanced",
			navScrollKey: "buddy-docs-nav-scroll",
			maxCollapsedCodeHeight: 420,
			largeTableMinimumRows: 8,
			headingSelector: ".md-typeset h2[id], .md-typeset h3[id], .md-typeset h4[id]",
			contentSelector: ".md-content__inner",
			navSelector: ".md-nav--primary .md-nav__list",
			tocSelector: ".md-nav--secondary",
			tableSelector: ".md-typeset table:not([data-buddy-no-filter])",
			codeSelector: ".md-typeset pre > code"
		},
		state: {
			pageReady: false,
			scrollTicking: false,
			resizeTicking: false
		},
		init: function()
		{
			if( document.documentElement.getAttribute( this.config.initializedAttribute ) ===
					"true" )
			{
				return;
			}
			document.documentElement.setAttribute( this.config.initializedAttribute, "true" );
			this.enhanceExternalLinks();
			this.customizeSearch();
			this.addReadingProgress();
			this.addScrollTopButton();
			this.addPageTools();
			this.addHeadingLinks();
			this.addTableFilters();
			this.addCodeLabels();
			this.addCodeToggles();
			this.addPagePathMetadata();
			this.restoreNavigationScroll();
			this.enhanceKeyboardFocus();
			this.enhanceApiReference();
			this.enhanceTocProgress();
			this.bindLifecycleEvents();
			this.state.pageReady = true;
			this.updateReadingProgress();
			this.updateScrollTopVisibility();
			this.updateTocProgress();
		},
		bindLifecycleEvents: function()
		{
			const self = this;
			window.addEventListener( "scroll", function()
			{
				if( !self.state.scrollTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateScrollTopVisibility();
						self.updateTocProgress();
						self.state.scrollTicking = false;
					} );
					self.state.scrollTicking = true;
				}
			}, { passive: true } );
			window.addEventListener( "resize", function()
			{
				if( !self.state.resizeTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateTocProgress();
						self.state.resizeTicking = false;
					} );
					self.state.resizeTicking = true;
				}
			}, { passive: true } );
			document.addEventListener( "click", function( event )
			{
				self.handleDocumentClick( event );
			} );
			document.addEventListener( "keydown", function( event )
			{
				self.handleKeyboardShortcuts( event );
			} );
			window.addEventListener( "beforeunload", function()
			{
				self.saveNavigationScroll();
			} );
			if( typeof document$ !== "undefined" && document$ && typeof document$.subscribe ===
					"function" )
			{
				document$.subscribe( function()
				{
					document.documentElement.removeAttribute( self.config.initializedAttribute );
					setTimeout( function()
					{
						self.init();
					}, 25 );
				} );
			}
		},
		handleDocumentClick: function( event )
		{
			const target = event.target;
			if( !target )
			{
				return;
			}
			if( target.closest && target.closest( "#" + this.config.scrollTopId ) )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( target.closest && target.closest( "[data-buddy-copy-heading]" ) )
			{
				event.preventDefault();
				this.copyHeadingLink( target.closest( "[data-buddy-copy-heading]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-buddy-copy-page]" ) )
			{
				event.preventDefault();
				this.copyPageLink( target.closest( "[data-buddy-copy-page]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-buddy-print-page]" ) )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( target.closest && target.closest( "[data-buddy-toggle-code]" ) )
			{
				event.preventDefault();
				this.toggleCodeBlock( target.closest( "[data-buddy-toggle-code]" ) );
			}
		},
		handleKeyboardShortcuts: function( event )
		{
			const key = ( event.key || "" ).toLowerCase();
			if( event.altKey && key === "t" )
			{
				event.preventDefault();
				this.scrollToTop();
			}
			if( event.altKey && key === "p" )
			{
				event.preventDefault();
				window.print();
			}
			if( event.altKey && key === "l" )
			{
				event.preventDefault();
				this.copyCurrentPageToClipboard();
			}
		},
		enhanceExternalLinks: function()
		{
			const links = document.querySelectorAll( ".md-typeset a[href]" );
			const currentHost = window.location.host;
			links.forEach( function( link )
			{
				try
				{
					const url = new URL( link.href, window.location.href );
					if( url.host && url.host !== currentHost )
					{
						link.setAttribute( "target", "_blank" );
						link.setAttribute( "rel", "noopener noreferrer" );
						link.classList.add( "buddy-external-link" );
						if( !link.querySelector( ".buddy-external-indicator" ) )
						{
							const indicator = document.createElement( "span" );
							indicator.className = "buddy-external-indicator";
							indicator.setAttribute( "aria-hidden", "true" );
							indicator.textContent = " ↗";
							link.appendChild( indicator );
						}
					}
				}
				catch( error )
				{
					return;
				}
			} );
		},
		customizeSearch: function()
		{
			const searchInputs = document.querySelectorAll( "input.md-search__input" );
			searchInputs.forEach( function( input )
			{
				input.setAttribute( "placeholder", "Search Buddy docs..." );
				input.setAttribute( "aria-label", "Search Buddy documentation" );
			} );
		},
		addReadingProgress: function()
		{
			if( document.getElementById( this.config.progressId ) )
			{
				return;
			}
			const progress = document.createElement( "div" );
			progress.id = this.config.progressId;
			progress.setAttribute( "aria-hidden", "true" );
			progress.innerHTML = "<span></span>";
			document.body.appendChild( progress );
		},
		updateReadingProgress: function()
		{
			const progress = document.querySelector( "#" + this.config.progressId + " span" );
			if( !progress )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			const scrollTop = window.scrollY || document.documentElement.scrollTop;
			let maxScroll = document.documentElement.scrollHeight - window.innerHeight;
			if( content )
			{
				const rect = content.getBoundingClientRect();
				const contentTop = rect.top + scrollTop;
				const contentHeight = Math.max( content.offsetHeight, 1 );
				const contentScroll = Math.min( Math.max( scrollTop - contentTop, 0 ),
						contentHeight );
				const percent = Math.min( Math.max( contentScroll / contentHeight, 0 ), 1 );
				progress.style.width = ( percent * 100 ).toFixed( 2 ) + "%";
				return;
			}
			if( maxScroll <= 0 )
			{
				maxScroll = 1;
			}
			progress.style.width =
					Math.min( Math.max( ( scrollTop / maxScroll ) * 100, 0 ), 100 ).toFixed( 2 ) +
					"%";
		},
		addScrollTopButton: function()
		{
			if( document.getElementById( this.config.scrollTopId ) )
			{
				return;
			}
			const button = document.createElement( "button" );
			button.id = this.config.scrollTopId;
			button.type = "button";
			button.className = "buddy-scroll-top";
			button.setAttribute( "aria-label", "Scroll to top" );
			button.setAttribute( "title", "Scroll to top (Alt+T)" );
			button.innerHTML = "↑";
			document.body.appendChild( button );
		},
		updateScrollTopVisibility: function()
		{
			const button = document.getElementById( this.config.scrollTopId );
			if( !button )
			{
				return;
			}
			if( ( window.scrollY || document.documentElement.scrollTop ) > 420 )
			{
				button.classList.add( "is-visible" );
			}
			else
			{
				button.classList.remove( "is-visible" );
			}
		},
		scrollToTop: function()
		{
			window.scrollTo( {
				top: 0,
				behavior: "smooth"
			} );
		},
		addPageTools: function()
		{
			if( document.getElementById( this.config.pageToolsId ) )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const tools = document.createElement( "div" );
			tools.id = this.config.pageToolsId;
			tools.className = "buddy-page-tools";
			tools.innerHTML = [
				"<button type=\"button\" data-buddy-copy-page title=\"Copy page link\" aria-label=\"Copy page link\">Copy link</button>",
				"<button type=\"button\" data-buddy-print-page title=\"Print page\" aria-label=\"Print page\">Print</button>"
			].join( "" );
			title.insertAdjacentElement( "afterend", tools );
		},
		copyPageLink: function( button )
		{
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		copyCurrentPageToClipboard: function()
		{
			const button = document.querySelector( "[data-buddy-copy-page]" );
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		addHeadingLinks: function()
		{
			const headings = document.querySelectorAll( this.config.headingSelector );
			headings.forEach( function( heading )
			{
				if( heading.querySelector( "." + BuddyDocs.config.headingLinkClass ) )
				{
					return;
				}
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = BuddyDocs.config.headingLinkClass;
				button.setAttribute( "data-buddy-copy-heading", heading.id );
				button.setAttribute( "aria-label", "Copy link to " + heading.textContent.trim() );
				button.setAttribute( "title", "Copy section link" );
				button.textContent = "§";
				heading.appendChild( button );
			} );
		},
		copyHeadingLink: function( button )
		{
			const id = button.getAttribute( "data-buddy-copy-heading" );
			if( !id )
			{
				return;
			}
			const url = window.location.origin + window.location.pathname + window.location.search +
					"#" + encodeURIComponent( id );
			this.copyTextToClipboard( url, button, "Copied", "§" );
		},
		copyTextToClipboard: function( text, button, successText, defaultText )
		{
			const updateButton = function()
			{
				if( !button )
				{
					return;
				}
				const previous = button.textContent;
				button.textContent = successText || "Copied";
				setTimeout( function()
				{
					button.textContent = defaultText || previous;
				}, 1400 );
			};
			if( navigator.clipboard && typeof navigator.clipboard.writeText === "function" )
			{
				navigator.clipboard.writeText( text ).then( updateButton ).catch( function()
				{
					BuddyDocs.fallbackCopyText( text );
					updateButton();
				} );
				return;
			}
			this.fallbackCopyText( text );
			updateButton();
		},
		fallbackCopyText: function( text )
		{
			const textarea = document.createElement( "textarea" );
			textarea.value = text;
			textarea.setAttribute( "readonly", "readonly" );
			textarea.style.position = "fixed";
			textarea.style.top = "-9999px";
			textarea.style.left = "-9999px";
			document.body.appendChild( textarea );
			textarea.select();
			try
			{
				document.execCommand( "copy" );
			}
			catch( error )
			{
				return;
			}
			finally
			{
				document.body.removeChild( textarea );
			}
		},
		addTableFilters: function()
		{
			const tables = document.querySelectorAll( this.config.tableSelector );
			tables.forEach( function( table, index )
			{
				if( table.getAttribute( "data-buddy-filtered" ) === "true" )
				{
					return;
				}
				const tbody = table.querySelector( "tbody" );
				if( !tbody )
				{
					return;
				}
				const rows = Array.prototype.slice.call( tbody.querySelectorAll( "tr" ) );
				if( rows.length < BuddyDocs.config.largeTableMinimumRows )
				{
					return;
				}
				table.setAttribute( "data-buddy-filtered", "true" );
				const wrapper = document.createElement( "div" );
				wrapper.className = "buddy-table-tools";
				const input = document.createElement( "input" );
				input.type = "search";
				input.className = BuddyDocs.config.tableFilterClass;
				input.placeholder = "Filter table...";
				input.setAttribute( "aria-label", "Filter table " + ( index + 1 ) );
				const count = document.createElement( "span" );
				count.className = "buddy-table-count";
				count.textContent = rows.length + " rows";
				wrapper.appendChild( input );
				wrapper.appendChild( count );
				table.parentNode.insertBefore( wrapper, table );
				input.addEventListener( "input", function()
				{
					BuddyDocs.filterTable( table, input.value, count );
				} );
			} );
		},
		filterTable: function( table, query, countElement )
		{
			const normalizedQuery = ( query || "" ).toLowerCase().trim();
			const rows = Array.prototype.slice.call( table.querySelectorAll( "tbody tr" ) );
			let visible = 0;
			rows.forEach( function( row )
			{
				const text = row.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					row.style.display = "";
					visible += 1;
				}
				else
				{
					row.style.display = "none";
				}
			} );
			if( countElement )
			{
				countElement.textContent = visible + " / " + rows.length + " rows";
			}
		},
		addCodeLabels: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-buddy-labeled" ) === "true" )
				{
					return;
				}
				const language = BuddyDocs.detectCodeLanguage( code );
				if( !language )
				{
					return;
				}
				pre.setAttribute( "data-buddy-labeled", "true" );
				const label = document.createElement( "div" );
				label.className = BuddyDocs.config.codeLabelClass;
				label.textContent = language;
				pre.insertAdjacentElement( "beforebegin", label );
			} );
		},
		detectCodeLanguage: function( code )
		{
			const className = code.className || "";
			const match = className.match( /language-([a-zA-Z0-9_+-]+)/ );
			if( match && match[ 1 ] )
			{
				return this.formatLanguageName( match[ 1 ] );
			}
			const text = code.textContent.trim();
			if( /^site_name:|^theme:|^plugins:|^nav:/m.test( text ) )
			{
				return "YAML";
			}
			if( /^(python|py)\s/i.test( text ) || /def\s+\w+\(|class\s+\w+/.test( text ) )
			{
				return "Python";
			}
			if( /^mkdocs\s|^python\s|-m\s+/.test( text ) )
			{
				return "Shell";
			}
			if( /^\{[\s\S]*\}$/.test( text ) )
			{
				return "JSON";
			}
			if( /^#\s|^##\s|```/.test( text ) )
			{
				return "Markdown";
			}
			return "";
		},
		formatLanguageName: function( language )
		{
			const map = {
				py: "Python",
				python: "Python",
				ps1: "PowerShell",
				powershell: "PowerShell",
				bash: "Shell",
				sh: "Shell",
				shell: "Shell",
				yaml: "YAML",
				yml: "YAML",
				json: "JSON",
				md: "Markdown",
				markdown: "Markdown",
				html: "HTML",
				css: "CSS",
				js: "JavaScript",
				javascript: "JavaScript"
			};
			const key = String( language || "" ).toLowerCase();
			return map[ key ] || key.toUpperCase();
		},
		addCodeToggles: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-buddy-toggle-ready" ) === "true" )
				{
					return;
				}
				pre.setAttribute( "data-buddy-toggle-ready", "true" );
				if( pre.scrollHeight <= BuddyDocs.config.maxCollapsedCodeHeight + 80 )
				{
					return;
				}
				pre.classList.add( "buddy-code-collapsed" );
				pre.style.maxHeight = BuddyDocs.config.maxCollapsedCodeHeight + "px";
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = BuddyDocs.config.codeToggleClass;
				button.setAttribute( "data-buddy-toggle-code", "collapsed" );
				button.textContent = "Show full code";
				pre.insertAdjacentElement( "afterend", button );
			} );
		},
		toggleCodeBlock: function( button )
		{
			const pre = button.previousElementSibling;
			if( !pre || pre.tagName.toLowerCase() !== "pre" )
			{
				return;
			}
			const state = button.getAttribute( "data-buddy-toggle-code" );
			if( state === "collapsed" )
			{
				pre.classList.remove( "buddy-code-collapsed" );
				pre.style.maxHeight = "";
				button.setAttribute( "data-buddy-toggle-code", "expanded" );
				button.textContent = "Collapse code";
			}
			else
			{
				pre.classList.add( "buddy-code-collapsed" );
				pre.style.maxHeight = this.config.maxCollapsedCodeHeight + "px";
				button.setAttribute( "data-buddy-toggle-code", "collapsed" );
				button.textContent = "Show full code";
			}
		},
		addPagePathMetadata: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || content.querySelector( ".buddy-page-path" ) )
			{
				return;
			}
			const h1 = content.querySelector( "h1" );
			if( !h1 )
			{
				return;
			}
			const path = window.location.pathname
					.replace( /\/$/, "" )
					.split( "/" )
					.filter( Boolean )
					.slice( -4 )
					.join( " / " );
			if( !path )
			{
				return;
			}
			const meta = document.createElement( "div" );
			meta.className = "buddy-page-path";
			meta.textContent = "Docs path: " + path;
			h1.insertAdjacentElement( "afterend", meta );
		},
		saveNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				window.sessionStorage.setItem( this.config.navScrollKey,
						String( nav.scrollTop || 0 ) );
			}
			catch( error )
			{
				return;
			}
		},
		restoreNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				const value = window.sessionStorage.getItem( this.config.navScrollKey );
				if( value !== null )
				{
					nav.scrollTop = parseInt( value, 10 ) || 0;
				}
			}
			catch( error )
			{
				return;
			}
		},
		enhanceKeyboardFocus: function()
		{
			document.body.addEventListener( "keydown", function( event )
			{
				if( event.key === "Tab" )
				{
					document.body.classList.add( "buddy-keyboard-mode" );
				}
			} );
			document.body.addEventListener( "mousedown", function()
			{
				document.body.classList.remove( "buddy-keyboard-mode" );
			} );
		},
		enhanceApiReference: function()
		{
			const apiContainers = document.querySelectorAll(
					".doc.doc-object, .doc-class, .doc-function, .doc-method, .doc-attribute" );
			apiContainers.forEach( function( container )
			{
				if( container.getAttribute( "data-buddy-api-enhanced" ) === "true" )
				{
					return;
				}
				container.setAttribute( "data-buddy-api-enhanced", "true" );
				const heading = container.querySelector( "h2, h3, h4, h5" );
				if( !heading || heading.querySelector( ".buddy-api-badge" ) )
				{
					return;
				}
				const badge = document.createElement( "span" );
				badge.className = "buddy-api-badge";
				if( container.className.indexOf( "doc-class" ) !== -1 )
				{
					badge.textContent = "class";
				}
				else if( container.className.indexOf( "doc-method" ) !== -1 )
				{
					badge.textContent = "method";
				}
				else if( container.className.indexOf( "doc-function" ) !== -1 )
				{
					badge.textContent = "function";
				}
				else if( container.className.indexOf( "doc-attribute" ) !== -1 )
				{
					badge.textContent = "attribute";
				}
				else
				{
					badge.textContent = "api";
				}
				heading.appendChild( badge );
			} );
		},
		enhanceTocProgress: function()
		{
			const toc = document.querySelector( this.config.tocSelector );
			if( !toc || toc.getAttribute( "data-buddy-toc-enhanced" ) === "true" )
			{
				return;
			}
			toc.setAttribute( "data-buddy-toc-enhanced", "true" );
			const marker = document.createElement( "div" );
			marker.className = "buddy-toc-marker";
			marker.setAttribute( "aria-hidden", "true" );
			toc.appendChild( marker );
		},
		updateTocProgress: function()
		{
			const headings = Array.prototype.slice.call(
					document.querySelectorAll( this.config.headingSelector ) );
			if( headings.length === 0 )
			{
				return;
			}
			let activeHeading = headings[ 0 ];
			const offset = 120;
			headings.forEach( function( heading )
			{
				const rect = heading.getBoundingClientRect();
				if( rect.top <= offset )
				{
					activeHeading = heading;
				}
			} );
			const tocLinks = document.querySelectorAll( this.config.tocSelector + " a[href^='#']" );
			const activeId = activeHeading
			                 ? activeHeading.id
			                 : "";
			tocLinks.forEach( function( link )
			{
				const href = decodeURIComponent(
						( link.getAttribute( "href" ) || "" ).replace( /^#/, "" ) );
				if( href === activeId )
				{
					link.classList.add( "buddy-toc-active" );
				}
				else
				{
					link.classList.remove( "buddy-toc-active" );
				}
			} );
		},
		addMermaidGuard: function()
		{
			const blocks = document.querySelectorAll( "code.language-mermaid" );
			blocks.forEach( function( block )
			{
				block.setAttribute( "data-buddy-mermaid-detected", "true" );
			} );
		}
	};
	
	function ready( callback )
	{
		if( document.readyState === "loading" )
		{
			document.addEventListener( "DOMContentLoaded", callback );
		}
		else
		{
			callback();
		}
	}
	
	ready( function()
	{
		BuddyDocs.init();
	} );
} )();