/*! js-cookie v3.0.0 | MIT */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):(e=e||self,function(){var n=e.Cookies,r=e.Cookies=t();r.noConflict=function(){return e.Cookies=n,r}}())}(this,(function(){"use strict";function e(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)e[r]=n[r]}return e}var t={read:function(e){return e.replace(/(%[\dA-F]{2})+/gi,decodeURIComponent)},write:function(e){return encodeURIComponent(e).replace(/%(2[346BF]|3[AC-F]|40|5[BDE]|60|7[BCD])/g,decodeURIComponent)}};return function n(r,o){function i(t,n,i){if("undefined"!=typeof document){"number"==typeof(i=e({},o,i)).expires&&(i.expires=new Date(Date.now()+864e5*i.expires)),i.expires&&(i.expires=i.expires.toUTCString()),t=encodeURIComponent(t).replace(/%(2[346B]|5E|60|7C)/g,decodeURIComponent).replace(/[()]/g,escape),n=r.write(n,t);var c="";for(var u in i)i[u]&&(c+="; "+u,!0!==i[u]&&(c+="="+i[u].split(";")[0]));return document.cookie=t+"="+n+c}}return Object.create({set:i,get:function(e){if("undefined"!=typeof document&&(!arguments.length||e)){for(var n=document.cookie?document.cookie.split("; "):[],o={},i=0;i<n.length;i++){var c=n[i].split("="),u=c.slice(1).join("=");'"'===u[0]&&(u=u.slice(1,-1));try{var f=t.read(c[0]);if(o[f]=r.read(u,f),e===f)break}catch(e){}}return e?o[e]:o}},remove:function(t,n){i(t,"",e({},n,{expires:-1}))},withAttributes:function(t){return n(this.converter,e({},this.attributes,t))},withConverter:function(t){return n(e({},this.converter,t),this.attributes)}},{attributes:{value:Object.freeze(o)},converter:{value:Object.freeze(r)}})}(t,{path:"/"})}));

window.SEMICOLON_cookieInit = function( $cookieEl ){

	$cookieEl = $cookieEl.filter(':not(.customjs)');

	if( $cookieEl.length < 1 ){
		return true;
	}

	let $cookieBar = $('.gdpr-settings'),
		elSpeed		= $cookieBar.attr('data-speed') || 300,
		elExpire	= $cookieBar.attr('data-expire') || 30,
		elDelay		= $cookieBar.attr('data-delay') || 1500,
		elPersist	= $cookieBar.attr('data-persistent'),
		elDirection	= 'bottom',
		elHeight	= $cookieBar.outerHeight() + 100,
		elWidth		= $cookieBar.outerWidth() + 100,
		elProp		= {},
		elSize,
		elSettings	= $('.gdpr-cookie-settings'),
		elSwitches	= elSettings.find('[data-cookie-name]');

	if( elPersist == 'true' ) {
		Cookies.set('websiteUsesCookies', '');
	}

	if( $cookieBar.hasClass('gdpr-settings-sm') && $cookieBar.hasClass('gdpr-settings-right') ) {
		elDirection = 'right';
	} else if( $cookieBar.hasClass('gdpr-settings-sm') ) {
		elDirection = 'left';
	}

	if( elDirection == 'left' ) {
		elSize	= -elWidth;
		elProp	= { 'left': elSize, 'right': 'auto' };
		elProp.marginLeft = '1rem';
	} else if( elDirection == 'right' ) {
		elSize	= -elWidth;
		elProp	= { 'right': elSize, 'left': 'auto' };
		elProp.marginRight = '1rem';
	} else {
		elSize	= -elHeight;
		elProp[elDirection] = elSize;
	}

	$cookieBar.css( elProp );

	if( Cookies.get('websiteUsesCookies') != 'yesConfirmed' ) {
		elProp[elDirection] = 0;
		elProp.opacity = 1;
		setTimeout( function(){
			$cookieBar.css( elProp );
		}, Number( elDelay ) );
	}

	$('.gdpr-accept').off( 'click' ).on( 'click', function(){
		elProp[elDirection] = elSize;
		elProp.opacity = 0;
		$cookieBar.css( elProp );
		Cookies.set('websiteUsesCookies', 'yesConfirmed', { expires: Number( elExpire ) });
		return false;
	});

	elSwitches.each( function(){
		let elSwitch = $(this),
			elCookie = elSwitch.attr( 'data-cookie-name' ),
			getCookie = Cookies.get( elCookie );

		if( typeof getCookie !== 'undefined' && getCookie == '1' ) {
			elSwitch.prop( 'checked', true );
		} else {
			elSwitch.prop( 'checked', false )
		}
	});

	$('.gdpr-save-cookies').off( 'click' ).on( 'click', function(){
		elSwitches.each( function(){
			let elSwitch = $(this),
				elCookie = elSwitch.attr( 'data-cookie-name' );

			if( elSwitch.prop( 'checked' ) ) {
				Cookies.set( elCookie, '1', { expires: Number( elExpire ) });
			} else {
				Cookies.remove( elCookie, '' );
			}
		});

		if( elSettings.parents( '.mfp-content' ).length > 0 ) {
			$.magnificPopup.close();
		}

		setTimeout( function(){
			window.location.reload();
		}, 500);

		return false;
	});

	$('.gdpr-container').each( function(){
		let element = $(this),
			elCookie = element.attr('data-cookie-name'),
			elContent = element.attr('data-cookie-content'),
			elContentAjax = element.attr('data-cookie-content-ajax'),
			getCookie = Cookies.get( elCookie );

		if( typeof getCookie !== 'undefined' && getCookie == '1' ) {
			element.addClass('gdpr-content-active');
			if( elContentAjax ) {
				element.load( elContentAjax );
			} else {
				if( elContent ) {
					element.append( elContent );
				}
			}
			SEMICOLON.initialize.resizeVideos({ parent: element });
		} else {
			element.addClass('gdpr-content-blocked');
		}
	});

};

