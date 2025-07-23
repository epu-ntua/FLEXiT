window.onload = function () {
    const hideTryOut = () => {
        document.querySelectorAll('.try-out').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.execute-wrapper').forEach(el => el.style.display = 'none');
    };

    // Wait for Swagger UI to load
    const observer = new MutationObserver(hideTryOut);
    observer.observe(document.body, { childList: true, subtree: true });
    hideTryOut();
};
