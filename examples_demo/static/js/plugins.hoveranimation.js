window.scwHoverAnimationPlugin = window.scwHoverAnimationPlugin || {};

window.SEMICOLON_hoverAnimationInit = function( $hoverAnimationEl ){

	$hoverAnimationEl = $hoverAnimationEl.filter(':not(.customjs)');

	if( $hoverAnimationEl.length < 1 ){
		return true;
	}

	$hoverAnimationEl.each(function(){
		let element			= $(this),
			elAnimate		= element.attr( 'data-hover-animate' ),
			elAnimateOut	= element.attr( 'data-hover-animate-out' ) || 'fadeOut',
			elSpeed			= element.attr( 'data-hover-speed' ) || 600,
			elDelay			= element.attr( 'data-hover-delay' ),
			elParent		= element.attr( 'data-hover-parent' ),
			elReset			= element.attr( 'data-hover-reset' ) || 'false';

		element.addClass( 'not-animated' );

		if( !elParent ) {
			if( element.parents( '.bg-overlay' ).length > 0 ) {
				elParent = element.parents( '.bg-overlay' );
			} else {
				elParent = element;
			}
		} else {
			if( elParent == 'self' ) {
				elParent = element;
			} else {
				elParent = element.parents( elParent );
			}
		}

		let elDelayT = 0;

		if( elDelay ) {
			elDelayT = Number( elDelay );
		}

		if( elSpeed ) {
			element.css({ 'animation-duration': Number( elSpeed ) + 'ms' });
		}

		let t, x;

		elParent.hover( function(){

			clearTimeout( x );
			t = setTimeout( function(){
					element.addClass( 'not-animated' ).removeClass( elAnimateOut + ' not-animated' ).addClass( elAnimate + ' animated' );
				}, elDelayT );

		}, function(){

			element.addClass( 'not-animated' ).removeClass( elAnimate + ' not-animated' ).addClass( elAnimateOut + ' animated' );
			if( elReset == 'true' ) {
				x = setTimeout( function(){
					element.removeClass( elAnimateOut + ' animated' ).addClass( 'not-animated' );
				}, Number( elSpeed ) );
			}
			clearTimeout( t );

		});
	});

};
