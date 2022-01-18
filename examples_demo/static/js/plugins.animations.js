window.scwAnimationsPlugin = window.scwAnimationsPlugin || {};

window.SEMICOLON_animationsInit = function( $animationEl ){

	$animationEl = $animationEl.filter(':not(.customjs)');

	if( $animationEl.length < 1 ){
		return true;
	}

	let SELECTOR			= '[data-animate]',
		ANIMATE_CLASS_NAME	= 'animated';

	let isAnimated = function(element) {
		element.classList.contains(ANIMATE_CLASS_NAME)
	};

	let intersectionObserver = new IntersectionObserver(
		function(entries, observer) {
			entries.forEach( function(entry) {

				let thisElement				= $(entry.target),
					elAnimation				= thisElement.attr('data-animate'),
					elAnimOut				= thisElement.attr('data-animate-out'),
					elAnimDelay				= thisElement.attr('data-delay'),
					elAnimDelayOut			= thisElement.attr('data-delay-out'),
					elAnimDelayTime			= 0,
					elAnimDelayOutTime		= 3000;

				if( thisElement.parents('.fslider.no-thumbs-animate').length > 0 ) { return true; }
				if( thisElement.parents('.swiper-slide').length > 0 ) { return true; }

				if( elAnimDelay ) { elAnimDelayTime = Number( elAnimDelay ) + 500; } else { elAnimDelayTime = 500; }
				if( elAnimOut && elAnimDelayOut ) { elAnimDelayOutTime = Number( elAnimDelayOut ) + elAnimDelayTime; }

				if( !thisElement.hasClass('animated') ) {
					thisElement.addClass('not-animated');
					if (entry.intersectionRatio > 0) {

						setTimeout(function() {
							thisElement.removeClass('not-animated').addClass( elAnimation + ' animated');
						}, elAnimDelayTime);

						if( elAnimOut ) {
							setTimeout( function() {
								thisElement.removeClass( elAnimation ).addClass( elAnimOut );
							}, elAnimDelayOutTime );
						}

					}
				}

				if( !thisElement.hasClass('not-animated') ) {
					observer.unobserve(entry.target);
				}
			});
		}
	);

	let elements = [].filter.call(
		document.querySelectorAll(SELECTOR),
		function(element){
			return !isAnimated(element, ANIMATE_CLASS_NAME);
		});

	elements.forEach( function(element){
		return intersectionObserver.observe(element)
	});

};

