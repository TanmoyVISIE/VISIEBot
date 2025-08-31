$(document).ready(function () {
    // Toggle chat button and chat box
    $('.chat-button').on('click', function () {
        $('.chat-button').css({ "display": "none" });
        $('.chat-box').css({ "visibility": "visible" });
    });

    $('.chat-box-header p').on('click', function () {
        $('.chat-button').css({ "display": "block" });
        $('.chat-box').css({ "visibility": "hidden" });
    });

    // Toggle modal
    $("#addExtra").on("click", function () {
        $(".modal").toggleClass("show-modal");
    });

    $(".modal-close-button").on("click", function () {
        $(".modal").toggleClass("show-modal");
    });

    // Chatbot functionality
    $('.send').on('click', sendMessage);
    $('#user-input').on('keypress', function (e) {
        if (e.which == 13) { // Enter key
            sendMessage();
        }
    });

    function sendMessage() {
        let userInput = $('#user-input').val().trim();
        if (userInput === "") return;

        // Add user's message to chat
        let userMessage = `
            <div class="chat-box-body-send">
                <p>${userInput}</p>
                <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            </div>`;
        $('.chat-box-body').append(userMessage);

        // Clear input
        $('#user-input').val("");

        // Send to backend via AJAX
        $.ajax({
            url: "/get",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ message: userInput }),
            success: function (response) {
                if (response.response) {
                    let botMessage = `
                        <div class="chat-box-body-receive">
                            <p>${response.response}</p>
                            <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>`;
                    $('.chat-box-body').append(botMessage);
                } else {
                    let errorMessage = `
                        <div class="chat-box-body-receive">
                            <p>Sorry, I couldn’t process your request. Please try again.</p>
                            <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>`;
                    $('.chat-box-body').append(errorMessage);
                }
                // Auto-scroll to bottom
                $('.chat-box-body').scrollTop($('.chat-box-body')[0].scrollHeight);
            },
            error: function (error) {
                console.error("Error:", error);
                let errorMessage = `
                    <div class="chat-box-body-receive">
                        <p>Sorry, there was an error. Please try again later.</p>
                        <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>`;
                $('.chat-box-body').append(errorMessage);
                $('.chat-box-body').scrollTop($('.chat-box-body')[0].scrollHeight);
            }
        });
    }
});