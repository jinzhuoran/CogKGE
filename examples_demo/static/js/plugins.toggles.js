window.scwTogglesPlugin = window.scwTogglesPlugin || {};

window.SEMICOLON_togglesInit = function( $toggleEl ){

	$toggleEl = $toggleEl.filter(':not(.customjs)');

	if( $toggleEl.length < 1 ){
		return true;
	}

	$toggleEl.each( function(){
		let element = $(this),
			elSpeed = element.attr('data-speed') || 300,
			elState = element.attr('data-state');

		if( elState != 'open' ){
			element.children('.toggle-content').hide();
		} else {
			element.addClass('toggle-active').children('.toggle-content').slideDown( Number(elSpeed) );
		}

		element.children('.toggle-header').off( 'click' ).on( 'click', function(){
			element.toggleClass('toggle-active').children('.toggle-content').slideToggle( Number(elSpeed) );
			return true;
		});
	});

};

