/*!
 * animsition v4.0.2
 * A simple and easy jQuery plugin for CSS animated page transitions.
 * http://blivesta.github.io/animsition
 * License : MIT
 * Author : blivesta (http://blivesta.com/)
 */
!function(t){"use strict";"function"==typeof define&&define.amd?define(["jquery"],t):"object"==typeof exports?module.exports=t(require("jquery")):t(jQuery)}(function(t){"use strict";var n=!1;t(window).on("load",function(){n=!0});var i="animsition",a={init:function(o){o=t.extend({inClass:"fade-in",outClass:"fade-out",inDuration:1500,outDuration:800,linkElement:".animsition-link",loading:!0,loadingParentElement:"body",loadingClass:"animsition-loading",loadingInner:"",timeout:!1,timeoutCountdown:5e3,onLoadEvent:!0,browser:["animation-duration","-webkit-animation-duration"],overlay:!1,overlayClass:"animsition-overlay-slide",overlayParentElement:"body",transition:function(t){window.location.href=t}},o),a.settings={timer:!1,data:{inClass:"animsition-in-class",inDuration:"animsition-in-duration",outClass:"animsition-out-class",outDuration:"animsition-out-duration",overlay:"animsition-overlay"},events:{inStart:"animsition.inStart",inEnd:"animsition.inEnd",outStart:"animsition.outStart",outEnd:"animsition.outEnd"}};var e=a.supportCheck.call(this,o);if(!e&&o.browser.length>0&&(!e||!this.length))return"console"in window||(window.console={},window.console.log=function(t){return t}),this.length||console.log("Animsition: Element does not exist on page."),e||console.log("Animsition: Does not support this browser."),a.destroy.call(this);var s=a.optionCheck.call(this,o);return s&&t("."+o.overlayClass).length<=0&&a.addOverlay.call(this,o),o.loading&&t("."+o.loadingClass).length<=0&&a.addLoading.call(this,o),this.each(function(){var e=this,s=t(this),r=t(window),l=t(document),d=s.data(i);d||(o=t.extend({},o),s.data(i,{options:o}),o.timeout&&a.addTimer.call(e),o.onLoadEvent&&(n?(a.settings.timer&&clearTimeout(a.settings.timer),a["in"].call(e)):r.on("load."+i,function(){a.settings.timer&&clearTimeout(a.settings.timer),a["in"].call(e)})),r.on("pageshow."+i,function(t){t.originalEvent.persisted&&a["in"].call(e)}),r.on("unload."+i,function(){}),l.on("click."+i,o.linkElement,function(n){n.preventDefault();var i=t(this),o=i.attr("href");2===n.which||n.metaKey||n.shiftKey||-1!==navigator.platform.toUpperCase().indexOf("WIN")&&n.ctrlKey?window.open(o,"_blank"):a.out.call(e,i,o)}))})},addOverlay:function(n){t(n.overlayParentElement).prepend('<div class="'+n.overlayClass+'"></div>')},addLoading:function(n){t(n.loadingParentElement).append('<div class="'+n.loadingClass+'">'+n.loadingInner+"</div>")},removeLoading:function(){var n=t(this),a=n.data(i).options,o=t(a.loadingParentElement).children("."+a.loadingClass);o.fadeOut().remove()},addTimer:function(){var n=this,o=t(this),e=o.data(i).options;a.settings.timer=setTimeout(function(){a["in"].call(n),t(window).off("load."+i)},e.timeoutCountdown)},supportCheck:function(n){var i=t(this),a=n.browser,o=a.length,e=!1;0===o&&(e=!0);for(var s=0;o>s;s++)if("string"==typeof i.css(a[s])){e=!0;break}return e},optionCheck:function(n){var i,o=t(this);return i=n.overlay||o.data(a.settings.data.overlay)?!0:!1},animationCheck:function(n,a,o){var e=t(this),s=e.data(i).options,r=typeof n,l=!a&&"number"===r,d=a&&"string"===r&&n.length>0;return l||d?n=n:a&&o?n=s.inClass:!a&&o?n=s.inDuration:a&&!o?n=s.outClass:a||o||(n=s.outDuration),n},"in":function(){var n=this,o=t(this),e=o.data(i).options,s=o.data(a.settings.data.inDuration),r=o.data(a.settings.data.inClass),l=a.animationCheck.call(n,s,!1,!0),d=a.animationCheck.call(n,r,!0,!0),u=a.optionCheck.call(n,e),c=o.data(i).outClass;e.loading&&a.removeLoading.call(n),c&&o.removeClass(c),u?a.inOverlay.call(n,d,l):a.inDefault.call(n,d,l)},inDefault:function(n,i){var o=t(this);o.css({"animation-duration":i+"ms"}).addClass(n).trigger(a.settings.events.inStart).animateCallback(function(){o.removeClass(n).css({opacity:1}).trigger(a.settings.events.inEnd)})},inOverlay:function(n,o){var e=t(this),s=e.data(i).options;e.css({opacity:1}).trigger(a.settings.events.inStart),t(s.overlayParentElement).children("."+s.overlayClass).css({"animation-duration":o+"ms"}).addClass(n).animateCallback(function(){e.trigger(a.settings.events.inEnd)})},out:function(n,o){var e=this,s=t(this),r=s.data(i).options,l=n.data(a.settings.data.outClass),d=s.data(a.settings.data.outClass),u=n.data(a.settings.data.outDuration),c=s.data(a.settings.data.outDuration),m=l?l:d,g=u?u:c,f=a.animationCheck.call(e,m,!0,!1),v=a.animationCheck.call(e,g,!1,!1),h=a.optionCheck.call(e,r);s.data(i).outClass=f,h?a.outOverlay.call(e,f,v,o):a.outDefault.call(e,f,v,o)},outDefault:function(n,o,e){var s=t(this),r=s.data(i).options;s.css({"animation-duration":o+1+"ms"}).addClass(n).trigger(a.settings.events.outStart).animateCallback(function(){s.trigger(a.settings.events.outEnd),r.transition(e)})},outOverlay:function(n,o,e){var s=this,r=t(this),l=r.data(i).options,d=r.data(a.settings.data.inClass),u=a.animationCheck.call(s,d,!0,!0);t(l.overlayParentElement).children("."+l.overlayClass).css({"animation-duration":o+1+"ms"}).removeClass(u).addClass(n).trigger(a.settings.events.outStart).animateCallback(function(){r.trigger(a.settings.events.outEnd),l.transition(e)})},destroy:function(){return this.each(function(){var n=t(this);t(window).off("."+i),n.css({opacity:1}).removeData(i)})}};t.fn.animateCallback=function(n){var i="animationend webkitAnimationEnd";return this.each(function(){var a=t(this);a.on(i,function(){return a.off(i),n.call(this)})})},t.fn.animsition=function(n){return a[n]?a[n].apply(this,Array.prototype.slice.call(arguments,1)):"object"!=typeof n&&n?void t.error("Method "+n+" does not exist on jQuery."+i):a.init.apply(this,arguments)}});

window.SEMICOLON_pageTransitionInit = function( $wrapperEl ){

	let $body		= $('body'),
		$wrapper	= $('#wrapper');

	if( $body.hasClass('no-transition') ) { return true; }
	if( !$body.hasClass('page-transition') ) { return true; }

	if( !$().animsition ) {
		$body.addClass('no-transition');
		console.log('pageTransition: Animsition not Defined.');
		return true;
	}

	window.onpageshow = function(event) {
		if(event.persisted) {
			window.location.reload();
		}
	};

	let elAnimIn				= $body.attr('data-animation-in') || 'fadeIn',
		elAnimOut				= $body.attr('data-animation-out') || 'fadeOut',
		elSpeedIn				= $body.attr('data-speed-in') || 1500,
		elSpeedOut				= $body.attr('data-speed-out') || 800,
		elTimeoutActive			= false,
		elTimeout				= $body.attr('data-loader-timeout'),
		elLoader				= $body.attr('data-loader'),
		elLoaderColor			= $body.attr('data-loader-color'),
		elLoaderHtml			= $body.attr('data-loader-html'),
		elLoaderAppend			= '',
		elLoaderBefore			= '<div class="css3-spinner">',
		elLoaderAfter			= '</div>',
		elLoaderBg				= '',
		elLoaderBorder			= '',
		elLoaderBgClass			= '',
		elLoaderBorderClass		= '',
		elLoaderBgClass2		= '',
		elLoaderBorderClass2	= '';

	if( !elTimeout ) {
		elTimeoutActive = false;
		elTimeout = false;
	} else {
		elTimeoutActive = true;
		elTimeout = Number(elTimeout);
	}

	if( elLoaderColor ) {
		if( elLoaderColor == 'theme' ) {
			elLoaderBgClass		= ' bg-color';
			elLoaderBorderClass	= ' border-color';
			elLoaderBgClass2		= ' class="bg-color"';
			elLoaderBorderClass2	= ' class="border-color"';
		} else {
			elLoaderBg		= ' style="background-color:'+ elLoaderColor +';"';
			elLoaderBorder	= ' style="border-color:'+ elLoaderColor +';"';
		}
	}

	if( elLoader == '2' ) {
		elLoaderAppend = '<div class="css3-spinner-flipper'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	} else if( elLoader == '3' ) {
		elLoaderAppend = '<div class="css3-spinner-double-bounce1'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-double-bounce2'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	} else if( elLoader == '4' ) {
		elLoaderAppend = '<div class="css3-spinner-rect1'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-rect2'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-rect3'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-rect4'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-rect5'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	} else if( elLoader == '5' ) {
		elLoaderAppend = '<div class="css3-spinner-cube1'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-cube2'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	} else if( elLoader == '6' ) {
		elLoaderAppend = '<div class="css3-spinner-scaler'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	} else if( elLoader == '7' ) {
		elLoaderAppend = '<div class="css3-spinner-grid-pulse"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '8' ) {
		elLoaderAppend = '<div class="css3-spinner-clip-rotate"><div'+ elLoaderBorderClass2 + elLoaderBorder +'></div></div>';
	} else if( elLoader == '9' ) {
		elLoaderAppend = '<div class="css3-spinner-ball-rotate"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '10' ) {
		elLoaderAppend = '<div class="css3-spinner-zig-zag"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '11' ) {
		elLoaderAppend = '<div class="css3-spinner-triangle-path"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '12' ) {
		elLoaderAppend = '<div class="css3-spinner-ball-scale-multiple"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '13' ) {
		elLoaderAppend = '<div class="css3-spinner-ball-pulse-sync"><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div><div'+ elLoaderBgClass2 + elLoaderBg +'></div></div>';
	} else if( elLoader == '14' ) {
		elLoaderAppend = '<div class="css3-spinner-scale-ripple"><div'+ elLoaderBorderClass2 + elLoaderBorder +'></div><div'+ elLoaderBorderClass2 + elLoaderBorder +'></div><div'+ elLoaderBorderClass2 + elLoaderBorder +'></div></div>';
	} else {
		elLoaderAppend = '<div class="css3-spinner-bounce1'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-bounce2'+ elLoaderBgClass +'"'+ elLoaderBg +'></div><div class="css3-spinner-bounce3'+ elLoaderBgClass +'"'+ elLoaderBg +'></div>';
	}

	if( !elLoaderHtml ) {
		elLoaderHtml = elLoaderAppend;
	}

	elLoaderHtml = elLoaderBefore + elLoaderHtml + elLoaderAfter;

	$wrapper.css({ 'opacity': 1 });

	$wrapper.animsition({
		inClass: elAnimIn,
		outClass: elAnimOut,
		inDuration: Number(elSpeedIn),
		outDuration: Number(elSpeedOut),
		linkElement: 'body:not(.device-md):not(.device-sm):not(.device-xs) .primary-menu:not(.on-click) .menu-link:not([target="_blank"]):not([href*="#"]):not([data-lightbox]):not([href^="mailto"]):not([href^="tel"]):not([href^="sms"]):not([href^="call"])',
		loading: true,
		loadingParentElement: 'body',
		loadingClass: 'page-transition-wrap',
		loadingInner: elLoaderHtml,
		timeout: elTimeoutActive,
		timeoutCountdown: elTimeout,
		onLoadEvent: true,
		browser: [ 'animation-duration', '-webkit-animation-duration'],
		overlay: false,
		overlayClass: 'animsition-overlay-slide',
		overlayParentElement: 'body'
	});

};

