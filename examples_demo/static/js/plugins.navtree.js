window.scwNavTreePlugin = window.scwNavTreePlugin || {};

window.SEMICOLON_navtreeInit = function( $navTreeEl ){

	$navTreeEl = $navTreeEl.filter(':not(.customjs)');

	if( $navTreeEl.length < 1 ){
		return true;
	}

	$navTreeEl.each( function(){
		let element		= $(this),
			elSpeed		= element.attr('data-speed') || 250,
			elEasing	= element.attr('data-easing') || 'swing';

		element.find( 'ul li:has(ul)' ).addClass('sub-menu');
		element.find( 'ul li:has(ul) > a' ).append( ' <i class="icon-angle-down"></i>' );

		if( element.hasClass('on-hover') ){
			element.find( 'ul li:has(ul):not(.active)' ).hover( function(e){
				$(this).children('ul').stop(true, true).slideDown( Number(elSpeed), elEasing);
			}, function(){
				$(this).children('ul').delay(250).slideUp( Number(elSpeed), elEasing);
			});
		} else {
			element.find( 'ul li:has(ul) > a' ).off( 'click' ).on( 'click', function(e){
				let childElement = $(this);
				element.find( 'ul li' ).not(childElement.parents()).removeClass('active');
				childElement.parent().children('ul').slideToggle( Number(elSpeed), elEasing, function(){
					$(this).find('ul').hide();
					$(this).find('li.active').removeClass('active');
				});
				element.find( 'ul li > ul' ).not(childElement.parent().children('ul')).not(childElement.parents('ul')).slideUp( Number(elSpeed), elEasing );
				childElement.parent('li:has(ul)').toggleClass('active');
				e.preventDefault();
			});
		}
	});

};

