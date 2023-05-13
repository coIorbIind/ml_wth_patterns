jQuery(document).ready(function ($) {
    $('#my_image').change(function(){
        const file = this.files[0];
        console.log(file);
        if (file){
          let reader = new FileReader();
          reader.onload = function(event){
            $('#my_image_preview').attr('src', event.target.result);
          }
          reader.readAsDataURL(file);
        }
      });

    $(function () {
        $('input[type=submit]').click(function (event) {
            event.preventDefault();

            var formData = new FormData();
            formData.append('file', $("#my_image")[0].files[0]);

            // send request
            $.ajax({
                type: 'POST',
                url: 'http://127.0.0.1:8000/api/v1/model/predict',
                data: formData,
                processData: false,
                contentType: false,
                success: function (msg) {
                    $('#prediction').text(`Это ${msg.prediction}`);
                }
            });
        });
    });
});
