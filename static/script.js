// Find output DOM associated to the DOM element passed as parameter
function findOutputForSlider( element ) {
    var idVal = element.id;
    outputs = document.getElementsByTagName( 'output' );
    for( var i = 0; i < outputs.length; i++ ) {
      if ( outputs[ i ].htmlFor == idVal )
        return outputs[ i ];
    }
 }
 
 function getSliderOutputPosition( slider ) {
   // Update output position
   var newPlace,
       minValue;
 
   var style = window.getComputedStyle( slider, null );
   // Measure width of range input
   sliderWidth = parseInt( style.getPropertyValue( 'width' ), 10 );
 
   // Figure out placement percentage between left and right of input
   if ( !slider.getAttribute( 'min' ) ) {
     minValue = 0;
   } else {
     minValue = slider.getAttribute( 'min' );
   }
   var newPoint = ( slider.value - minValue ) / ( slider.getAttribute( 'max' ) - minValue );
 
   // Prevent bubble from going beyond left or right (unsupported browsers)
   if ( newPoint < 0 ) {
     newPlace = 0;
   } else if ( newPoint > 1 ) {
     newPlace = sliderWidth;
   } else {
     newPlace = sliderWidth * newPoint;
   }
 
   return {
     'position': newPlace + 'px'
   }
 }
 
 document.addEventListener( 'DOMContentLoaded', function () {
   // Get all document sliders
   var sliders = document.querySelectorAll( 'input[type="range"].slider' );
   [].forEach.call( sliders, function ( slider ) {
     var output = findOutputForSlider( slider );
     if ( output ) {
       if ( slider.classList.contains( 'has-output-tooltip' ) ) {
         // Get new output position
         var newPosition = getSliderOutputPosition( slider );
 
         // Set output position
         output.style[ 'left' ] = newPosition.position;
       }
 
       // Add event listener to update output when slider value change
       slider.addEventListener( 'input', function( event ) {
         if ( event.target.classList.contains( 'has-output-tooltip' ) ) {
           // Get new output position
           var newPosition = getSliderOutputPosition( event.target );
 
           // Set output position
           output.style[ 'left' ] = newPosition.position;
         }
 
         // Update output with slider value
         output.value = event.target.value;
       } );
     }
   } );
 } );

$(function() {
    $('#gen-form').submit(function(e) {
        e.preventDefault();
        $.ajax({
            type: "POST",
            url: "http://0.0.0.0:8080/predict",
            dataType: "json",
            data: JSON.stringify(getInputValues()),
            beforeSend: function(data) {
                $('#generate-text').addClass("is-loading");
                $('#generate-text').prop("disabled", true);
            },
            success: function(data) {
                $('#generate-text').removeClass("is-loading");
                $('#generate-text').prop("disabled", false);
                $('#tutorial').remove();
                var gentext = data.text;
                if ($("#prompt").length & $("#prompt").val() != '') {
                    var pattern = new RegExp('^' + $("#prompt").val(), 'g');
                    var gentext = gentext.replace(pattern, '<strong>' + $("#prompt").val() + '</strong>');
                }

                var gentext = gentext.replace(/\n\n/g, "<div><br></div>").replace(/\n/g, "<div></div>");
                var html = '<div class=\"gen-box\">' + gentext + '</div>';
                $(html).appendTo('#model-output').hide().fadeIn("slow");
            },
            error: function(jqXHR, textStatus, errorThrown) {
                $('#generate-text').removeClass("is-loading");
                $('#generate-text').prop("disabled", false);
                $('#tutorial').remove();
                var html = '<div class="gen-box warning">Attention, an error has occurred! Please try again.</div>';
                $(html).appendTo('#model-output').hide().fadeIn("slow");
            }
        });
    });
    $('#clear-text').click(function(e) {
        $('#model-output').text('')
    });

});

function getInputValues() {
    var inputs = {};
    $("select, textarea, input").each(function() {
        inputs[$(this).attr('id')] = $(this).val();
    });
    return inputs;
}
