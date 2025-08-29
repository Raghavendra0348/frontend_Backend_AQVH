function toggleForm() {
        const loginForm = document.querySelector('.login-form');
        const signupForm = document.querySelector('.signup-form');

        loginForm.classList.toggle('active');
        signupForm.classList.toggle('active');

        if (loginForm.classList.contains('active')) {
                loginForm.style.transform = 'translateX(-100%)';
                signupForm.style.transform = 'translateX(0)';
                signupForm.style.opacity = '1';
                loginForm.style.opacity = '0';
        } else {
                loginForm.style.transform = 'translateX(0)';
                signupForm.style.transform = 'translateX(100%)';
                loginForm.style.opacity = '1';
                signupForm.style.opacity = '0';
        }
}
function togglePassword() {
    const passwordField = document.getElementById("loginPassword");
    const eyeIcon = document.getElementById("eyeIcon");

    if (passwordField.type === "password") {
        passwordField.type = "text";
        eyeIcon.classList.remove("fa-eye");
        eyeIcon.classList.add("fa-eye-slash");
    } else {
        passwordField.type = "password";
        eyeIcon.classList.remove("fa-eye-slash");
        eyeIcon.classList.add("fa-eye");
    }
}
