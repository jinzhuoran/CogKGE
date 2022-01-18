window.scwDataHeightsPlugin = window.scwDataHeightsPlugin || {};

window.SEMICOLON_dataHeightsInit = function( $dataHeightsEl ){

	if( $dataHeightsEl.length < 1 ){
		return true;
	}

	let $dataHeightXs	= $('[data-height-xs]'),
		$dataHeightSm	= $('[data-height-sm]'),
		$dataHeightMd	= $('[data-height-md]'),
		$dataHeightLg	= $('[data-height-lg]'),
		$dataHeightXl	= $('[data-height-xl]'),
		$body			= $('body');

	if( $dataHeightXs.length > 0 ) {
		$dataHeightXs.each( function(){
			let element = $(this),
				elementHeight = element.attr('data-height-xs');

			if( $body.hasClass('device-xs') ) {
				if( elementHeight != '' ) { element.css( 'height', elementHeight ); }
			}
		});
	}

	if( $dataHeightSm.length > 0 ) {
		$dataHeightSm.each( function(){
			let element = $(this),
				elementHeight = element.attr('data-height-sm');

			if( $body.hasClass('device-sm') ) {
				if( elementHeight != '' ) { element.css( 'height', elementHeight ); }
			}
		});
	}

	if( $dataHeightMd.length > 0 ) {
		$dataHeightMd.each( function(){
			let element = $(this),
				elementHeight = element.attr('data-height-md');

			if( $body.hasClass('device-md') ) {
				if( elementHeight != '' ) { element.css( 'height', elementHeight ); }
			}
		});
	}

	if( $dataHeightLg.length > 0 ) {
		$dataHeightLg.each( function(){
			let element = $(this),
				elementHeight = element.attr('data-height-lg');

			if( $body.hasClass('device-lg') ) {
				if( elementHeight != '' ) { element.css( 'height', elementHeight ); }
			}
		});
	}

	if( $dataHeightXl.length > 0 ) {
		$dataHeightXl.each( function(){
			let element = $(this),
				elementHeight = element.attr('data-height-xl');

			if( $body.hasClass('device-xl') ) {
				if( elementHeight != '' ) { element.css( 'height', elementHeight ); }
			}
		});
	}

};

