<?php

header ("Content-Type:text/css");

/** ===============================================================
 *
 *      Edit your Color Configurations below:
 *      You should only enter 6-Digits HEX Colors.
 *
 ================================================================== */

$color = "#1ABC9C"; // Change your Color Here

/** ===============================================================
 *
 *      Do not Edit anything below this line if you do not know
 *      what you are trying to do..!
 *
 ================================================================== */

function checkhexcolor($color) {

	return preg_match('/^#[a-f0-9]{6}$/i', $color);

}

/** ===============================================================
 *
 *      Primary Color Scheme
 *
 ================================================================== */

if( isset( $_GET[ 'color' ] ) AND $_GET[ 'color' ] != '' ) {
	$color = "#" . $_GET[ 'color' ];
}

if( !$color OR !checkhexcolor( $color ) ) {
	$color = "#1ABC9C";
}

?>


/* ----------------------------------------------------------------
	Colors

	Replace the HEX Code with your Desired Color HEX
-----------------------------------------------------------------*/


::selection { background: <?php echo $color; ?>; }

::-moz-selection { background: <?php echo $color; ?>; }

::-webkit-selection { background: <?php echo $color; ?>; }


a,
h1 > span:not(.nocolor):not(.badge),
h2 > span:not(.nocolor):not(.badge),
h3 > span:not(.nocolor):not(.badge),
h4 > span:not(.nocolor):not(.badge),
h5 > span:not(.nocolor):not(.badge),
h6 > span:not(.nocolor):not(.badge),
.header-extras li .he-text span,
.menu-item:hover > .menu-link,
.menu-item.current > .menu-link,
.dark .menu-item:hover > .menu-link,
.dark .menu-item.current > .menu-link,
.top-cart-item-desc a:hover,
.top-cart-action .top-checkout-price,
.breadcrumb a:hover,
.grid-filter li:not(.activeFilter) a:hover,
.portfolio-desc h3 a:hover,
#portfolio-navigation a:hover,
.entry-title h2 a:hover,
.entry-title h3 a:hover,
.entry-title h4 a:hover,
.post-timeline .entry:hover .entry-timeline,
.post-timeline .entry:hover .timeline-divider,
.comment-content .comment-author a:hover,
.product-title h3 a:hover,
.single-product .product-title h2 a:hover,
.product-price ins,
.single-product .product-price,
.process-steps li.active h5,
.process-steps li.ui-tabs-active h5,
.tab-nav-lg li.ui-tabs-active a,
.team-title span,
.btn-link,
.page-link,
.page-link:hover,
.page-link:focus,
.fbox-plain .fbox-icon i,
.fbox-plain .fbox-icon img,
.fbox-border .fbox-icon i,
.fbox-border .fbox-icon img,
.iconlist > li [class^="icon-"]:first-child,
.dark .menu-item:hover > .menu-link,
.dark .menu-item.current > .menu-link,
.dark .top-cart-item-desc a:hover,
.dark .breadcrumb a:hover,
.dark .portfolio-desc h3 a:hover,
.dark #portfolio-navigation a:hover,
.dark .entry-title h2 a:hover,
.dark .entry-title h3 a:hover,
.dark .entry-title h4 a:hover,
.dark .product-title h3 a:hover,
.dark .single-product .product-title h2 a:hover,
.dark .product-price ins,
.dark .tab-nav-lg li.ui-tabs-active a { color: <?php echo $color; ?>; }

.color,
.h-text-color:hover,
a.h-text-color:hover,
.grid-filter.style-3 li.activeFilter a,
.faqlist li a:hover,
.tagcloud a:hover,
.nav-tree li:hover > a,
.nav-tree li.current > a,
.nav-tree li.active > a { color: <?php echo $color; ?> !important; }

.top-cart-number::before,
#page-menu-wrap,
.page-menu-nav,
.control-solid .flex-control-nav li:hover a,
.control-solid .flex-control-nav li a.flex-active,
.grid-filter li.activeFilter a,
.grid-filter.style-4 li.activeFilter a::after,
.grid-shuffle:hover,
.entry-link:hover,
.button,
.button.button-dark:hover,
.button.button-3d:hover,
.fbox-icon i,
.fbox-icon img,
.fbox-effect.fbox-dark .fbox-icon i:hover,
.fbox-effect.fbox-dark:hover .fbox-icon i,
.fbox-border.fbox-effect.fbox-dark .fbox-icon i::after,
.i-rounded:hover,
.i-circled:hover,
.tab-nav.tab-nav2 li.ui-state-active a,
.testimonial .flex-control-nav li a,
.skills li .progress,
.owl-carousel .owl-dots .owl-dot,
#gotoTop:hover,
input.switch-toggle-round:checked + label::before,
input.switch-toggle-flat:checked + label,
input.switch-toggle-flat:checked + label::after,
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus,
input.switch-toggle-round:checked + label::before,
input.switch-toggle-flat:checked + label,
input.switch-toggle-flat:checked + label::after,
.dark .entry-link:hover,
.dark .fbox-effect.fbox-dark .fbox-icon i:hover,
.dark .fbox-effect.fbox-dark:hover .fbox-icon i,
.dark .fbox-border.fbox-effect.fbox-dark .fbox-icon i::after,
.dark .i-rounded:hover,
.dark .i-circled:hover,
.dark .tab-nav.tab-nav2 li.ui-state-active a,
.dark #gotoTop:hover,
.dark input.switch-toggle-round:checked + label::before,
.dark input.switch-toggle-flat:checked + label,
.dark input.switch-toggle-flat:checked + label::after { background-color: <?php echo $color; ?>; }

.bg-color,
.bg-color #header-wrap,
.h-bg-color:hover,
.process-steps li.active a,
.process-steps li.ui-tabs-active a,
.sidenav > .ui-tabs-active > a,
.sidenav > .ui-tabs-active > a:hover,
.owl-carousel .owl-nav [class*=owl-]:hover,
.widget-filter-links li.active-filter span,
.page-item.active .page-link,
.page-link:hover,
.page-link:focus { background-color: <?php echo $color; ?> !important; }

.bootstrap-switch .bootstrap-switch-handle-on.bootstrap-switch-themecolor,
.bootstrap-switch .bootstrap-switch-handle-off.bootstrap-switch-themecolor,
.checkbox-style:checked + .checkbox-style-1-label::before,
.checkbox-style:checked + .checkbox-style-2-label::before,
.checkbox-style:checked + .checkbox-style-3-label::before,
.radio-style:checked + .radio-style-3-label::before { background: <?php echo $color; ?>; }

.irs-bar,
.irs-from,
.irs-to,
.irs-single,
.irs-handle > i:first-child,
.irs-handle.state_hover > i:first-child,
.irs-handle:hover > i:first-child { background-color: <?php echo $color; ?> !important; }

.top-cart-item-image:hover,
.grid-filter.style-3 li.activeFilter a,
.post-timeline .entry:hover .entry-timeline,
.post-timeline .entry:hover .timeline-divider,
.cart-product-thumbnail img:hover,
.fbox-outline .fbox-icon a,
.fbox-border .fbox-icon a,
.heading-block.border-color::after,
.page-item.active .page-link,
.page-link:focus,
.dark .cart-product-thumbnail img:hover { border-color: <?php echo $color; ?>; }

.border-color,
.process-steps li.active a,
.process-steps li.ui-tabs-active a,
.tagcloud a:hover,
.page-link:hover { border-color: <?php echo $color; ?> !important; }

.top-links-sub-menu,
.top-links-section,
.tabs-tb .tab-nav li.ui-tabs-active a,
.dark .top-links-sub-menu,
.dark .top-links-section,
.dark .tabs-tb .tab-nav li.ui-tabs-active a { border-top-color: <?php echo $color; ?>; }

.title-border-color::before,
.title-border-color::after,
.irs-from::after,
.irs-single::after,
.irs-to::after,
.irs-from::before,
.irs-to::before,
.irs-single::before { border-top-color: <?php echo $color; ?> !important; }

.title-block { border-left-color: <?php echo $color; ?>; }

.rtl .title-block {
	border-left-color: transparent;
	border-right-color: <?php echo $color; ?>;
}

.title-block-right { border-right-color: <?php echo $color; ?>; }

.rtl .title-block-right {
	border-right-color: transparent;
	border-left-color: <?php echo $color; ?>;
}

.more-link,
.tabs-bb .tab-nav li.ui-tabs-active a,
.title-bottom-border h1,
.title-bottom-border h2,
.title-bottom-border h3,
.title-bottom-border h4,
.title-bottom-border h5,
.title-bottom-border h6 { border-bottom-color: <?php echo $color; ?>; }

.fbox-effect.fbox-dark .fbox-icon i::after,
.dark .fbox-effect.fbox-dark .fbox-icon i::after { box-shadow: 0 0 0 2px <?php echo $color; ?>; }

.fbox-border.fbox-effect.fbox-dark .fbox-icon i:hover,
.fbox-border.fbox-effect.fbox-dark:hover .fbox-icon i,
.dark .fbox-border.fbox-effect.fbox-dark .fbox-icon i:hover,
.dark .fbox-border.fbox-effect.fbox-dark:hover .fbox-icon i { box-shadow: 0 0 0 1px <?php echo $color; ?>; }


@media (min-width: 992px) {

	.sub-menu-container .menu-item:hover > .menu-link,
	.mega-menu-style-2 .mega-menu-title > .menu-link:hover,
	.dark .mega-menu-style-2 .mega-menu-title:hover > .menu-link { color: <?php echo $color; ?>; }

	.style-3 .menu-container > .menu-item.current > .menu-link,
	.sub-title .menu-container > .menu-item:hover > .menu-link::after,
	.sub-title .menu-container > .menu-item.current > .menu-link::after,
	.page-menu-sub-menu,
	.dots-menu .page-menu-item.current > a,
	.dots-menu .page-menu-item div,
	.dark .style-3 .menu-container > .menu-item.current > .menu-link { background-color: <?php echo $color; ?>; }

	.style-4 .menu-container > .menu-item:hover > .menu-link,
	.style-4 .menu-container > .menu-item.current > .menu-link,
	.dots-menu.dots-menu-border .page-menu-item.current > a { border-color: <?php echo $color; ?>; }

	.sub-menu-container,
	.mega-menu-content,
	.style-6 .menu-container > .menu-item > .menu-link::after,
	.style-6 .menu-container > .menu-item.current > .menu-link::after,
	.top-cart-content,
	.dark .sub-menu-container,
	.dark .mega-menu-content,
	.dark .top-cart-content { border-top-color: <?php echo $color; ?>; }

	.dots-menu .page-menu-item div::after { border-left-color: <?php echo $color; ?>; }

	.rtl .dots-menu .page-menu-item div::after {
		border-left-color: transparent;
		border-right-color: <?php echo $color; ?>;
	}
}