window.onload = () => {                                    // execute on page load
	$('#sendbutton').click(() => {                         // execute when Get Prediction is clicked
		imagebox = $('#imagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://localhost:5000/detectObject", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);    // upload error log for no file upload
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					console.log(data);                    // success log for image upload
					bytestring = data['status']
					image = bytestring.split('\'')[1]
					imagebox.attr('src' , 'data:image/jpeg;base64,'+image)
				}
			});
		}
	});
};



function readUrl(input){
	imagebox = $('#imagebox')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			// console.log(e)
			
			imagebox.attr('src',e.target.result); 
			//imagebox.height(500);
			//imagebox.width(800);
			imagebox.height(800);
			imagebox.width(800);
		}
		reader.readAsDataURL(input.files[0]);
	}

	
}