window.scwPricingSwitcherPlugin = window.scwPricingSwitcherPlugin || {};

window.SEMICOLON_pricingSwitcherFn = function( checkbox, parent, pricing, defClass, actClass ) {
	parent.find('.pts-left,.pts-right').removeClass( actClass ).addClass( defClass );
	pricing.find('.pts-switch-content-left,.pts-switch-content-right').addClass('d-none');

	if( checkbox.filter(':checked').length > 0 ) {
		parent.find('.pts-right').removeClass( defClass ).addClass( actClass );
		pricing.find('.pts-switch-content-right').removeClass('d-none');
	} else {
		parent.find('.pts-left').removeClass( defClass ).addClass( actClass );
		pricing.find('.pts-switch-content-left').removeClass('d-none');
	}
};

window.SEMICOLON_pricingSwitcherInit = function( $pricingSwitcherEl ){

	$pricingSwitcherEl = $pricingSwitcherEl.filter(':not(.customjs)');

	if( $pricingSwitcherEl.length < 1 ){
		return true;
	}

	$pricingSwitcherEl.each( function(){
		var element		= $(this),
			elCheck		= element.find(':checkbox'),
			elParent	= $(this).parents('.pricing-tenure-switcher'),
			elDefClass	= $(this).attr('data-default-class') || 'text-muted op-05',
			elActClass	= $(this).attr('data-active-class') || 'fw-bold',
			elPricing	= $( elParent.attr('data-container') );

			console.log( elDefClass );

		SEMICOLON_pricingSwitcherFn( elCheck, elParent, elPricing, elDefClass, elActClass );

		elCheck.on( 'change', function(){
			SEMICOLON_pricingSwitcherFn( elCheck, elParent, elPricing, elDefClass, elActClass );
		});
	});

};

