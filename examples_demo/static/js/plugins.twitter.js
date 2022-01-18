/**
 * Twitter Feed Fetcher
 */

function sm_format_twitter(twitters) {
  var statusHTML = [];
  for (var i=0; i<twitters.length; i++){
	var username = twitters[i].user.screen_name;
	var name = twitters[i].user.name;
	var username_avatar = twitters[i].user.profile_image_url_https;
	var status = twitters[i].text.replace(/((https?|s?ftp|ssh)\:\/\/[^"\s\<\>]*[^.,;'">\:\s\<\>\)\]\!])/g, function(url) {
	  return '<a href="'+url+'" target="_blank">'+url+'</a>';
	}).replace(/\B@([_a-z0-9]+)/ig, function(reply) {
	  return  reply.charAt(0)+'<a href="https://twitter.com/'+reply.substring(1)+'" target="_blank">'+reply.substring(1)+'</a>';
	});
	statusHTML.push('<li><i class="icon-twitter"></i><a href="https://twitter.com/'+username+'" class="twitter-avatar" target="_blank"><img src="'+username_avatar+'" alt="'+name+'" title="'+name+'"></a><div><span>'+status+'</span><small><a href="https://twitter.com/'+username+'/statuses/'+twitters[i].id_str+'" target="_blank">'+relative_time(twitters[i].created_at)+'</a></small></div></li>');
  }
  return statusHTML.join('');
}

function sm_format_twitter2(twitters) {
  var statusHTML = [];
  for (var i=0; i<twitters.length; i++){
	var username = twitters[i].user.screen_name;
	var status = twitters[i].text.replace(/((https?|s?ftp|ssh)\:\/\/[^"\s\<\>]*[^.,;'">\:\s\<\>\)\]\!])/g, function(url) {
	  return '<a href="'+url+'" target="_blank">'+url+'</a>';
	}).replace(/\B@([_a-z0-9]+)/ig, function(reply) {
	  return  reply.charAt(0)+'<a href="https://twitter.com/'+reply.substring(1)+'" target="_blank">'+reply.substring(1)+'</a>';
	});
	statusHTML.push('<div class="slide"><span>'+status+'</span><small><a href="https://twitter.com/'+username+'/statuses/'+twitters[i].id_str+'" target="_blank">'+relative_time(twitters[i].created_at)+'</a></small></div>');
  }
  return statusHTML.join('');
}

function sm_format_twitter3(twitters) {
  var statusHTML = [];
  for (var i=0; i<twitters.length; i++){
	var username = twitters[i].user.screen_name;
	var status = twitters[i].text.replace(/((https?|s?ftp|ssh)\:\/\/[^"\s\<\>]*[^.,;'">\:\s\<\>\)\]\!])/g, function(url) {
	  return '<a href="'+url+'" target="_blank">'+url+'</a>';
	}).replace(/\B@([_a-z0-9]+)/ig, function(reply) {
	  return  reply.charAt(0)+'<a href="https://twitter.com/'+reply.substring(1)+'" target="_blank">'+reply.substring(1)+'</a>';
	});
	statusHTML.push('<div class="slide"><div class="testi-content"><p>'+status+'</p><div class="testi-meta"><span><a href="https://twitter.com/'+username+'/statuses/'+twitters[i].id_str+'" target="_blank">'+relative_time(twitters[i].created_at)+'</a></span></div></div></div>');
  }
  return statusHTML.join('');
}

function relative_time(time_value) {
  var values = time_value.split(" ");
  time_value = values[1] + " " + values[2] + ", " + values[5] + " " + values[3];
  var parsed_date = Date.parse(time_value);
  var relative_to = (arguments.length > 1) ? arguments[1] : new Date();
  var delta = parseInt((relative_to.getTime() - parsed_date) / 1000);
  delta = delta + (relative_to.getTimezoneOffset() * 60);

  if (delta < 60) {
	return 'less than a minute ago';
  } else if(delta < 120) {
	return 'about a minute ago';
  } else if(delta < (60*60)) {
	return (parseInt(delta / 60)).toString() + ' minutes ago';
  } else if(delta < (120*60)) {
	return 'about an hour ago';
  } else if(delta < (24*60*60)) {
	return 'about ' + (parseInt(delta / 3600)).toString() + ' hours ago';
  } else if(delta < (48*60*60)) {
	return '1 day ago';
  } else {
	return (parseInt(delta / 86400)).toString() + ' days ago';
  }
}

window.SEMICOLON_twitterFeedInit = function( $twitterFeedEl ){

	if( $twitterFeedEl.length < 1 ){
		return true;
	}

	$twitterFeedEl.each(function() {
		let element		= $(this),
			elUser		= element.attr('data-username') || 'twitter',
			elCount		= element.attr('data-count') || 3,
			elLoader	= element.attr('data-loader') || 'include/twitter/tweets.php';

		$.getJSON( elLoader + '?username='+ elUser +'&count='+ Number( elCount ), function(tweets){
			if( element.hasClass('fslider') ) {
				element.find(".slider-wrap").html(sm_format_twitter3(tweets)).promise().done( function(){
					let timer = setInterval(function(){
						if( element.find('.slide').length > 1 ) {
							element.removeClass('customjs');
							setTimeout( function(){ SEMICOLON.widget.loadFlexSlider(); }, 500);
							clearInterval(timer);
						}
					}, 500);
				});
			} else {
				element.html(sm_format_twitter(tweets));
			}
		});
	});

};
