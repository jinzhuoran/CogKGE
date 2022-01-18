window.scwQuantityPlugin = window.scwQuantityPlugin || {};

window.SEMICOLON_quantityInit = function( $quantityEl ){

	$quantityEl = $quantityEl.filter(':not(.customjs)');

	if( $quantityEl.length < 1 ){
		return true;
	}

	$(".plus").off( 'click' ).on( 'click', function(){
		let element = $(this).parents('.quantity').find('.qty'),
			elValue = element.val(),
			elStep = element.attr('step') || 1,
			elMax = element.attr('max'),
			intRegex = /^\d+$/;

		if( elMax && ( Number(elValue) >= Number( elMax ) ) ) { return false; }

		if( intRegex.test( elValue ) ) {
			let elValuePlus = Number(elValue) + Number(elStep);
			element.val( elValuePlus ).change();
		} else {
			element.val( Number(elStep) ).change();
		}

		return false;
	});

	$(".minus").off( 'click' ).on( 'click', function(){
		let element = $(this).parents('.quantity').find('.qty'),
			elValue = element.val(),
			elStep = element.attr('step') || 1,
			elMin = element.attr('min'),
			intRegex = /^\d+$/;

		if( !elMin || elMin < 0 ) { elMin = 1; }

		if( intRegex.test( elValue ) ) {
			if( Number(elValue) > Number( elMin ) ) {
				let elValueMinus = Number(elValue) - Number(elStep);
				element.val( elValueMinus ).change();
			}
		} else {
			element.val( Number(elStep) ).change();
		}

		return false;
	});

};

