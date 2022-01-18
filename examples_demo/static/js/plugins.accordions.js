window.scwAccordionsPlugin = window.scwAccordionsPlugin || {};

window.SEMICOLON_accordionsInit = function( $accordionEl ){

	$accordionEl = $accordionEl.filter(':not(.customjs)');

	if( $accordionEl.length < 1 ){
		return true;
	}

	$accordionEl.each( function(){
		let element			= $(this),
			elState			= element.attr('data-state'),
			elActive		= element.attr('data-active') || 1,
			elActiveClass	= element.attr('data-active-class') || '',
			elCollapsible	= element.attr('data-collapsible') || 'false',
			windowHash		= location.hash,
			accActive;

		elActive = Number( elActive ) - 1;

		if( typeof windowHash !== 'undefined' && windowHash != '' ) {
			accActive = element.find('.accordion-header'+ windowHash);
			if( accActive.length > 0 ) {
				elActive = accActive.index() / 2;
			}
		}

		element.find('.accordion-content').hide();

		if( elState != 'closed' ) {
			element.find('.accordion-header:eq('+ Number(elActive) +')').addClass('accordion-active ' + elActiveClass).next().show();
		}

		element.find('.accordion-header').off( 'click' ).on( 'click', function(){
			if( $(this).next().is(':hidden') ) {
				element.find('.accordion-header').removeClass('accordion-active ' + elActiveClass).next().slideUp("normal");
				let clickTarget = $(this);
				$(this).toggleClass('accordion-active ' + elActiveClass, true).next().stop(true,true).slideDown("normal", function(){
					if( ( $('body').hasClass('device-sm') || $('body').hasClass('device-xs') ) && element.hasClass('scroll-on-open') ) {
						$('html,body').stop(true,true).animate({
							'scrollTop': clickTarget.offset().top - ( SEMICOLON.initialize.topScrollOffset() - 40 )
						}, 800, 'easeOutQuad' );
					}
				});
			} else {
				if( elCollapsible == 'true' ) {
					$(this).toggleClass('accordion-active ' + elActiveClass, false).next().stop(true,true).slideUp("normal");
				}
			}
			return false;
		});
	});

};

