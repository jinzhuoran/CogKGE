window.scwSubscribeFormPlugin = window.scwSubscribeFormPlugin || {};

window.SEMICOLON_subscribeFormInit = function( $subscribeForm ){

	$subscribeForm = $subscribeForm.filter(':not(.customjs)');

	if( $subscribeForm.length < 1 ){
		return true;
	}

	$subscribeForm.each( function(){
		let element = $(this),
			elAlert = element.attr('data-alert-type'),
			elLoader = element.attr('data-loader'),
			elResult = element.find('.widget-subscribe-form-result'),
			elRedirect = element.attr('data-redirect'),
			defButton, defButtonText, alertType;

		element.find('form').validate({
			submitHandler: function(form) {

				elResult.hide();

				if( elLoader == 'button' ) {
					defButton = $(form).find('button');
					defButtonText = defButton.html();

					defButton.html('<i class="icon-line-loader icon-spin nomargin"></i>');
				} else {
					$(form).find('.icon-email2').removeClass('icon-email2').addClass('icon-line-loader icon-spin');
				}

				$(form).ajaxSubmit({
					target: elResult,
					dataType: 'json',
					resetForm: true,
					success: function( data ) {
						if( elLoader == 'button' ) {
							defButton.html( defButtonText );
						} else {
							$(form).find('.icon-line-loader').removeClass('icon-line-loader icon-spin').addClass('icon-email2');
						}
						if( data.alert != 'error' && elRedirect ){
							window.location.replace( elRedirect );
							return true;
						}
						if( elAlert == 'inline' ) {
							if( data.alert == 'error' ) {
								alertType = 'alert-danger';
							} else {
								alertType = 'alert-success';
							}

							elResult.addClass( 'alert ' + alertType ).html( data.message ).slideDown( 400 );
						} else {
							elResult.attr( 'data-notify-type', data.alert ).attr( 'data-notify-msg', data.message ).html('');
							SEMICOLON.widget.notifications({ el: elResult });
						}
					}
				});
			}
		});

	});

};

