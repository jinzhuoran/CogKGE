window.scwProgressPlugin = window.scwProgressPlugin || {};

window.SEMICOLON_progressInit = function( $progressEl ){

	$progressEl = $progressEl.filter(':not(.customjs)');

	if( $progressEl.length < 1 ){
		return true;
	}

	$progressEl.each(function(){
		let element	= $(this),
			elBar	= element.parent('li'),
			elValue	= elBar.attr('data-percent');

		if( element.parent('.kv-upload-progress').length > 0 || element.children('.progress-bar').length > 0 ) {
			return true;
		}

		let observer = new IntersectionObserver( function(entries, observer){
			entries.forEach( function(entry){
				if (entry.isIntersecting) {
					if (!elBar.hasClass('skills-animated')) {
						SEMICOLON.widget.counter({
							el: element.find('.counter-instant')
						});
						elBar.find('.progress').css({width: elValue + "%"}).addClass('skills-animated');
					}
					observer.unobserve( entry.target );
				}
			});
		}, {rootMargin: '-50px'});
		observer.observe( elBar[0] );
	});

};
