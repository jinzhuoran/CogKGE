window.scwInstagramPlugin = window.scwInstagramPlugin || {};

window.SEMICOLON_instagramPhotosInit = function( $instagramPhotosEl ){

	if( $instagramPhotosEl.length < 1 ){
		return true;
	}

	$instagramPhotosEl.each(function() {
		let element		= $(this),
			elLimit		= element.attr('data-count') || 12,
			elLoader	= element.attr('data-loader') || 'include/instagram/instagram.php',
			elFetch		= element.attr('data-fetch-message') || 'Fetching Photos from Instagram...';

		if( Number( elLimit ) > 12 ) {
			elLimit = 12;
		}

		SEMICOLON_getInstagramPhotos( element, elLoader, elLimit, elFetch );
	});

};

window.SEMICOLON_getInstagramPhotos = function( element, loader, limit, fetchAlert ) {

	let newimages = '';

	element.after( '<div class="alert alert-warning instagram-widget-alert text-center"><div class="spinner-grow spinner-grow-sm mr-2" role="status"><span class="visually-hidden">Loading...</span></div> '+ fetchAlert +'</div>' );

	$.getJSON( loader, function( images ){

		if( images.length > 0 ) {
			element.parents().find( '.instagram-widget-alert' ).remove();
			let html = '';
			for (let i = 0; i < limit; i++) {
				if ( i === limit )
					continue;

				let photo = images[i],
					thumb = photo.media_url;
				if( photo.media_type === 'VIDEO' ) {
					thumb = photo.thumbnail_url;
				}
				element.append( '<a class="grid-item" href="'+ photo.permalink +'" target="_blank"><img src="'+ thumb +'" alt="Image"></a>' );

				// $.getJSON( 'https://graph.instagram.com/' + images[i].id + '?fields=media_url,permalink,media_type,thumbnail_url&access_token=' + accessToken, function( photo ){

				// });

			}
		}

		element.removeClass('customjs');
		setTimeout( function(){
			SEMICOLON.widget.gridInit();
			SEMICOLON.widget.masonryThumbs();
		}, 500);

	});

};
