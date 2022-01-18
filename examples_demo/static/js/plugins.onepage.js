window.scwOnePageModulePlugin = window.scwOnePageModulePlugin || {};

let $onePageMenu,
	windowEl		= $(window),
	windowWidth		= windowEl.width(),
	$body			= $('body'),
	$pageSectionEl,
	locationHash	= location.hash,
	onePageMenuActiveClass;

window.SEMICOLON_onePageModule = function( $onePageMenuEl ){

	$onePageMenu = $onePageMenuEl.filter(':not(.customjs)');

	if( $onePageMenu.length < 1 ){
		return true;
	}

	if( $('a[data-href="'+ locationHash +'"]').parents( '.one-page-menu' ).length > 0 ) {

		windowEl.on('beforeunload', function() {
			windowEl.scrollTop(0);
		});

		windowEl.scrollTop(0);

		let linkElement = $( locationHash );

		if( linkElement.length > 0 ) {
			setTimeout( function(){
				let linkElementOff	= linkElement.offset().top,
					linkElSettings = JSON.parse( linkElement.attr('data-onepage-settings') );

				$('html,body').stop(true, true).animate({
					'scrollTop': linkElementOff - Number( linkElSettings.offset )
				}, linkElSettings.speed, linkElSettings.easing );
			}, 1200);
		}

	}

	let onePageScrollInterval = setInterval( function(){
		if( 'pluginOnePageModuleReady' in scwEvents ) {
			windowEl.scrollEnd( function(){
				SEMICOLON_onePageScroller();
			}, 500 );
			clearInterval( onePageScrollInterval );
		}
	}, 1000 );

	$onePageMenu.each( function(){

		let	element		= $(this);

		onePageMenuActiveClass	= element.attr('data-active-class') || 'current';
		$onePageMenu	= element;

		if( element.hasClass('no-offset') ) {
			elOffset = 0;
		}

		let linkEls = element.find('a[data-href]');

		linkEls.each( function(){

			let linkEl			= $(this),
				linkElAnchor	= linkEl.attr('data-href'),
				linkElement		= $( linkElAnchor );

			if( linkElement.length > 0 ) {
				let linkElSettings = SEMICOLON_onePageSettings( linkElAnchor, $('a[data-href="'+ linkElAnchor +'"]') );
				setTimeout( function(){
					linkElement.attr( 'data-onepage-settings', JSON.stringify( linkElSettings ) );
					$pageSectionEl = $('[data-onepage-settings]');
				}, 1000);
			}

		});

		linkEls.off( 'click' ).on( 'click', function(){

			let linkEl			= $(this),
				linkElAnchor	= linkEl.attr('data-href'),
				linkElement		= $( linkElAnchor );

			if( linkElement.length > 0 ) {

				let linkElSettings = JSON.parse( linkElement.attr('data-onepage-settings') );

				element.find('li').removeClass( onePageMenuActiveClass );
				element.find('a[data-href="' + linkElAnchor + '"]').parent('li').addClass( onePageMenuActiveClass );

				if( windowWidth < 992 || $body.hasClass('overlay-menu') ) {
					$onePageMenuEl.parents('.primary-menu').filter( ':not(.mobile-menu-off-canvas)' ).find('.menu-container').stop(true, true).slideUp(200);
					$( '.primary-menu.mobile-menu-off-canvas .menu-container' ).toggleClass('d-block', false);
					$('#page-menu').toggleClass('page-menu-open', false);
					$body.toggleClass('primary-menu-open', false);
				}

				setTimeout( function(){
					let linkElementOff	= linkElement.offset().top;
					$('html,body').stop(true, true).animate({
						'scrollTop': linkElementOff - Number( linkElSettings.offset )
					}, Number(linkElSettings.speed), linkElSettings.easing);
				}, 250);
			}

			return false;
		});

	});

};

window.SEMICOLON_onePageScroller = function(){
	$onePageMenu.find('[data-href]').parent().removeClass( onePageMenuActiveClass );
	$onePageMenu.find('[data-href="#' + SEMICOLON_onePageCurrentSection() + '"]').parent().addClass( onePageMenuActiveClass );
};

window.SEMICOLON_onePageCurrentSection = function(){
	let currentOnePageSection;

	if( typeof $pageSectionEl === 'undefined' ) {
		return true;
	}

	$pageSectionEl.each(function(index) {
		let element = $(this),
			elSettings = JSON.parse( element.attr('data-onepage-settings') ),
			h = element.offset().top - elSettings.offset - 5,
			y = windowEl.scrollTop();

		if( ( y >= h ) && ( y < h + element.height() ) && ( element.attr('id') != currentOnePageSection ) ) {
			currentOnePageSection = element.attr('id');
		}
	});

	return currentOnePageSection;
};

window.SEMICOLON_onePageSettings = function( hash, element ) {

	if( typeof hash === 'undefined' || element.length < 1 ) {
		return true;
	}

	if( $( hash ).length < 1 ) {
		return true;
	}

	let settings	= {},
		child		= element,
		parent		= element.parents( '.one-page-menu' ),
		offsetTop	= $( hash ).offset().top,
		elSpeed		= child.attr( 'data-speed' ),
		elOffset	= child.attr( 'data-offset' ),
		elEasing	= child.attr( 'data-easing' );

	if( !elSpeed ) {
		elSpeed	= parent.attr( 'data-speed' ) || 1250;
	}

	if( !elEasing ) {
		elEasing	= parent.attr( 'data-easing' ) || 'easeOutQuad';
	}

	if( !elOffset ) {
		element		= parent;
		elOffset	= parent.attr( 'data-offset' ) || SEMICOLON.initialize.topScrollOffset();
	}

	let elOffsetXXL	= element.attr( 'data-offset-xxl' ),
		elOffsetXL	= element.attr( 'data-offset-xl' ),
		elOffsetLG	= element.attr( 'data-offset-lg' ),
		elOffsetMD	= element.attr( 'data-offset-md' ),
		elOffsetSM	= element.attr( 'data-offset-sm' ),
		elOffsetXS	= element.attr( 'data-offset-xs' );

	if( !elOffsetXXL ) { elOffsetXXL = Number(elOffset); }
	if( !elOffsetXL ) { elOffsetXL = Number(elOffsetXXL); }
	if( !elOffsetLG ) { elOffsetLG = Number(elOffsetXL); }
	if( !elOffsetMD ) { elOffsetMD = Number(elOffsetLG); }
	if( !elOffsetSM ) { elOffsetSM = Number(elOffsetMD); }
	if( !elOffsetXS ) { elOffsetXS = Number(elOffsetSM); }

	if( $body.hasClass('device-xxl') ) {
		elOffset = elOffsetXXL;
	} else if( $body.hasClass('device-xl') ) {
		elOffset = elOffsetXL;
	} else if( $body.hasClass('device-lg') ) {
		elOffset = elOffsetLG;
	} else if( $body.hasClass('device-md') ) {
		elOffset = elOffsetMD;
	} else if( $body.hasClass('device-sm') ) {
		elOffset = elOffsetSM;
	} else if( $body.hasClass('device-xs') ) {
		elOffset = elOffsetXS;
	}

	settings.offset = elOffset;

	settings.speed = elSpeed;
	settings.easing = elEasing;

	return settings;

};

