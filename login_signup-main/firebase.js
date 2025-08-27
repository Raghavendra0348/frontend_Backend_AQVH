import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-app.js";
import {
    getAuth,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signInWithPopup,
    GoogleAuthProvider,
    GithubAuthProvider,
    TwitterAuthProvider,
    updateProfile
} from "https://www.gstatic.com/firebasejs/10.7.0/firebase-auth.js";

// Firebase configuration
const firebaseConfig = {
 

    apiKey: "AIzaSyCPiwVC_iAXd9XEPq30OGL2_MV5KiSXUIg",
    authDomain: "sample-login-for-all.firebaseapp.com",
    projectId: "sample-login-for-all",
    storageBucket: "sample-login-for-all.appspot.app",
    messagingSenderId: "673683309349",
    appId: "1:673683309349:web:bc8e0f4991774d94a750cc"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Redirect on successful login/signup
function redirectToIndex(user) {
    const username = user.displayName || user.email;
    localStorage.setItem('username', username);
    window.location.href = "index_landing.html";
}

// Login handler
document.getElementById('loginForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value.trim();

    if (!email || !password) {
        alert("Please enter both email and password.");
        return;
    }

    signInWithEmailAndPassword(auth, email, password)
        .then(userCredential => {
            setTimeout(() => redirectToIndex(userCredential.user), 500);
        })
        .catch(error => alert(`Error: ${error.message}`));
});

// Signup handler
document.getElementById('signupForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const name = document.getElementById('signupName').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value.trim();

    if (!name || !email || !password) {
        alert("Please fill in all fields.");
        return;
    }

    createUserWithEmailAndPassword(auth, email, password)
        .then(userCredential => {
            const user = userCredential.user;

            updateProfile(user, { displayName: name })
                .then(() => {
                    localStorage.setItem('username', name);
                    alert('Signup Successful');
                    setTimeout(() => redirectToIndex(user), 500);
                })
                .catch(error => {
                    console.error("Error updating profile:", error);
                    alert(`Signup successful, but failed to update name: ${error.message}`);
                    setTimeout(() => redirectToIndex(user), 500);
                });
        })
        .catch(error => alert(`Error: ${error.message}`));
});

// Social login handler
window.socialLogin = function (provider) {
    let selectedProvider;

    switch (provider) {
        case 'Google':
            selectedProvider = new GoogleAuthProvider();
            selectedProvider.setCustomParameters({ prompt: 'select_account' });
            break;
        case 'GitHub':
            selectedProvider = new GithubAuthProvider();
            break;
        case 'Twitter':
            selectedProvider = new TwitterAuthProvider();
            break;
        default:
            alert("Unsupported provider");
            return;
    }

    signInWithPopup(auth, selectedProvider)
        .then(result => {
            alert(`Logged in with ${provider} successfully!`);
            setTimeout(() => redirectToIndex(result.user), 500);
        })
        .catch(error => alert(`Error: ${error.message}`));
};

import { signOut } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-auth.js";

function logout() {
    signOut(auth).then(() => {
        localStorage.clear();
        window.location.href = "login.html";
    });
}
