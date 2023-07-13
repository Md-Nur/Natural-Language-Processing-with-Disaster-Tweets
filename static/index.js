 // Mobile menu toggle functionality
 document.getElementById('mobile-menu-button').addEventListener('click', function () {
    let mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenu.classList.contains('hidden')) {
      mobileMenu.classList.remove('hidden');
    } else {
      mobileMenu.classList.add('hidden');
    }
  });