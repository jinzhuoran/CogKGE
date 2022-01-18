window.scwReadMorePlugin = window.scwReadMorePlugin || {};

window.SEMICOLON_readmoreInit = function( $readmoreEl ){

	$readmoreEl = $readmoreEl.filter(':not(.customjs)');

	if( $readmoreEl.length < 1 ){
		return true;
	}

	$readmoreEl.each( function(){
		let element		= $(this),
			elHeight	= element.outerHeight(),
			elSize		= element.attr('data-readmore-size') || '10rem',
			elSpeed		= element.attr('data-readmore-speed') || 500,
			elTrigger	= element.attr('data-readmore-trigger') || '.read-more-trigger',
			elTriggerO	= element.attr('data-readmore-trigger-open') || 'Read More',
			elTriggerC	= element.attr('data-readmore-trigger-close') || 'Read Less';

		elTrigger = element.find( elTrigger );
		elTrigger.html( elTriggerO );
		elSpeed = Number( elSpeed );

		element.addClass( 'read-more-wrap' ).css({ 'height': elSize, '-webkit-transition-duration': elSpeed + 'ms', 'transition-duration': elSpeed + 'ms' }).append('<div class="read-more-mask"></div>');

		let elMask		= element.find('.read-more-mask'),
			elMaskD		= element.attr('data-readmore-mask') || 'true',
			elMaskColor	= element.attr('data-readmore-maskcolor') || '#FFF',
			elMaskSize	= element.attr('data-readmore-masksize') || '100%';

		if( elMaskD == 'true' ) {
			elMask.css({ 'height': elMaskSize, 'background-image': 'linear-gradient( '+ SEMICOLON_HEXtoRGBA( elMaskColor, 0 ) +', '+ SEMICOLON_HEXtoRGBA( elMaskColor, 1 ) +' )' });
		} else {
			elMask.addClass('d-none');
		}

		elTrigger.off( 'click' ).on( 'click', function(){
			if( element.hasClass('read-more-wrap-open') ) {
				element.css({ 'height': elSize }).removeClass('read-more-wrap-open');
				setTimeout( function(){
					elTrigger.html( elTriggerO );
				}, elSpeed );
				if( elMaskD == 'true' ) {
					elMask.fadeIn( elSpeed );
				}
			} else {
				if( elTriggerC == 'false' ) {
					elTrigger.remove();
				}
				let elHeightN = elHeight + elTrigger.outerHeight();
				element.css({ 'height': elHeightN, 'overflow': '' }).addClass('read-more-wrap-open');
				setTimeout( function(){
					elTrigger.html( elTriggerC );
				}, elSpeed );
				if( elMaskD == 'true' ) {
					elMask.fadeOut( elSpeed );
				}
			}

			return false;
		});

	});

};

window.SEMICOLON_HEXtoRGBA = function( hex, op ){
	let c;
	if(/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)){
		c= hex.substring(1).split('');
		if(c.length== 3){
			c= [c[0], c[0], c[1], c[1], c[2], c[2]];
		}
		c= '0x'+c.join('');
		return 'rgba('+[(c>>16)&255, (c>>8)&255, c&255].join(',')+','+op+')';
	}
	console.log('Bad Hex');
};

