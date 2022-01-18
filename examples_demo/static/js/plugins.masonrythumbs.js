window.scwMasonryThumbsPlugin = window.scwMasonryThumbsPlugin || {};

window.SEMICOLON_masonryThumbsInit = function( $masonryThumbsEl ){

	if( $masonryThumbsEl.length < 1 ){
		return true;
	}

	let $body = $('body');

	$masonryThumbsEl.each( function(){
		let element			= $(this),
			elBig			= element.attr('data-big');

		element.children().css({ 'width': '' });

		let firstElementWidth = element.children().eq(0).outerWidth();

		element.filter('.has-init-isotope').isotope({
			masonry: {
				columnWidth: firstElementWidth
			}
		});

		if( elBig ) {
			elBig = elBig.split(",");
			let elBigNum = '',
				bigi = '';
			for( bigi = 0; bigi < elBig.length; bigi++ ){
				elBigNum = Number(elBig[bigi]) - 1;
				element.children().eq(elBigNum).addClass('grid-item-big');
			}
		}

		element.find('.grid-item-big').css({ width: firstElementWidth*2 + 'px' });

		setTimeout( function(){
			element.filter('.has-init-isotope').isotope( 'layout' );
		}, 500);
	});

};
