window.scwAjaxPortfolioPlugin = window.scwAjaxPortfolioPlugin || {};

let $portfolioAjaxItems			= $('.portfolio-ajax').find('.portfolio-item'),
	$portfolioDetails			= $('#portfolio-ajax-wrap'),
	$portfolioDetailsContainer	= $('#portfolio-ajax-container'),
	$portfolioAjaxLoader		= $('#portfolio-ajax-loader'),
	prevPostPortId				= '';

window.SEMICOLON_portfolioAjaxloadInit = function(){
	if( $('.portfolio-ajax').length < 1 ){
		return true;
	}

	$('.portfolio-ajax .portfolio-item a.portfolio-ajax-trigger').off( 'click' ).on( 'click', function(e) {
		let portPostId = $(this).parents('.portfolio-item').attr('id');
		if( !$(this).parents('.portfolio-item').hasClass('portfolio-active') ) {
			SEMICOLON_portfolioLoadItem(portPostId, prevPostPortId);
		}
		e.preventDefault();
	});
};

window.SEMICOLON_portfolionewNextPrev = function( portPostId ){
	let portNext = SEMICOLON_portfolioGetNextItem(portPostId);
	let portPrev = SEMICOLON_portfolioGetPrevItem(portPostId);
	$('#next-portfolio').attr('data-id', portNext);
	$('#prev-portfolio').attr('data-id', portPrev);
};

window.SEMICOLON_portfolioLoadItem = function( portPostId, prevPostPortId, getIt ){
	if(!getIt) { getIt = false; }
	let portNext = SEMICOLON_portfolioGetNextItem(portPostId);
	let portPrev = SEMICOLON_portfolioGetPrevItem(portPostId);
	if(getIt == false) {
		SEMICOLON_portfolioCloseItem();
		$portfolioAjaxLoader.fadeIn();
		let portfolioDataLoader = $('#' + portPostId).attr('data-loader');
		$portfolioDetailsContainer.load(portfolioDataLoader, { portid: portPostId, portnext: portNext, portprev: portPrev },
		function(){
			SEMICOLON_portfolioInitializeAjax(portPostId);
			SEMICOLON_portfolioOpenItem();
			$portfolioAjaxItems.removeClass('portfolio-active');
			$('#' + portPostId).addClass('portfolio-active');
		});
	}
};

window.SEMICOLON_portfolioCloseItem = function(){
	if( $portfolioDetails && $portfolioDetails.height() > 32 ) {
		$portfolioAjaxLoader.fadeIn();
		$portfolioDetails.find('#portfolio-ajax-single').fadeOut('600', function(){
			$(this).remove();
		});
		$portfolioDetails.removeClass('portfolio-ajax-opened');
	}
};

window.SEMICOLON_portfolioOpenItem = function(){
	let noOfImages = $portfolioDetails.find('img').length;
	let noLoaded = 0;

	if( noOfImages > 0 ) {
		$portfolioDetails.find('img').on('load', function(){
			noLoaded++;
			let topOffsetScroll = SEMICOLON.initialize.topScrollOffset();
			if(noOfImages === noLoaded) {
				$portfolioDetailsContainer.css({ 'display': 'block' });
				$portfolioDetails.addClass('portfolio-ajax-opened');
				$portfolioAjaxLoader.fadeOut();
				setTimeout(function(){
					SEMICOLON.widget.loadFlexSlider();
					SEMICOLON.initialize.lightbox({ 'parent': $portfolioDetails });
					SEMICOLON.initialize.resizeVideos();
					SEMICOLON.widget.masonryThumbs();
					$('html,body').stop(true).animate({
						'scrollTop': $portfolioDetails.offset().top - topOffsetScroll
					}, 900, 'easeOutQuad');
				},500);
			}
		});
	} else {
		let topOffsetScroll = SEMICOLON.initialize.topScrollOffset();
		$portfolioDetailsContainer.css({ 'display': 'block' });
		$portfolioDetails.addClass('portfolio-ajax-opened');
		$portfolioAjaxLoader.fadeOut();
		setTimeout(function(){
			SEMICOLON.widget.loadFlexSlider();
			SEMICOLON.initialize.lightbox({ 'parent': $portfolioDetails });
			SEMICOLON.initialize.resizeVideos();
			SEMICOLON.widget.masonryThumbs();
			$('html,body').stop(true).animate({
				'scrollTop': $portfolioDetails.offset().top - topOffsetScroll
			}, 900, 'easeOutQuad');
		},500);
	}
};

window.SEMICOLON_portfolioGetNextItem = function( portPostId ){
	let portNext = '';
	let hasNext = $('#' + portPostId).next();
	if(hasNext.length != 0) {
		portNext = hasNext.attr('id');
	}
	return portNext;
};

window.SEMICOLON_portfolioGetPrevItem = function( portPostId ){
	let portPrev = '';
	let hasPrev = $('#' + portPostId).prev();
	if(hasPrev.length != 0) {
		portPrev = hasPrev.attr('id');
	}
	return portPrev;
};

window.SEMICOLON_portfolioInitializeAjax = function( portPostId ){
	prevPostPortId = $('#' + portPostId);

	$('#next-portfolio, #prev-portfolio').off( 'click' ).on( 'click', function() {
		let portPostId = $(this).attr('data-id');
		$portfolioAjaxItems.removeClass('portfolio-active');
		$('#' + portPostId).addClass('portfolio-active');
		SEMICOLON_portfolioLoadItem(portPostId,prevPostPortId);
		return false;
	});

	$('#close-portfolio').off( 'click' ).on( 'click', function() {
		$portfolioDetailsContainer.fadeOut('600', function(){
			$portfolioDetails.find('#portfolio-ajax-single').remove();
		});
		$portfolioDetails.removeClass('portfolio-ajax-opened');
		$portfolioAjaxItems.removeClass('portfolio-active');
		return false;
	});
};

