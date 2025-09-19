/**
 * Speech Synthesis Utility for Curriculum Vitae
 * Handles text-to-speech functionality with fallback options
 */

class SpeechUtils {
    constructor() {
        this.isSupported = 'speechSynthesis' in window;
        this.isReady = false;
        this.voices = [];
        this.init();
    }

    init() {
        if (!this.isSupported) {
            console.warn('Speech Synthesis not supported in this browser');
            return;
        }

        // Wait for voices to be loaded
        if (speechSynthesis.getVoices().length > 0) {
            this.loadVoices();
        } else {
            speechSynthesis.addEventListener('voiceschanged', () => {
                this.loadVoices();
            });
        }

        // Ensure speech synthesis is ready after user interaction
        document.addEventListener('click', this.ensureReady.bind(this), { once: true });
        document.addEventListener('touchstart', this.ensureReady.bind(this), { once: true });
    }

    loadVoices() {
        this.voices = speechSynthesis.getVoices();
        this.isReady = true;
        console.log('Speech synthesis ready with', this.voices.length, 'voices');
    }

    ensureReady() {
        if (this.isSupported && !this.isReady) {
            // Try to trigger voice loading
            const utterance = new SpeechSynthesisUtterance('');
            utterance.volume = 0;
            speechSynthesis.speak(utterance);
            speechSynthesis.cancel();
        }
    }

    findBestVoice(lang = 'zh-CN') {
        if (!this.voices.length) return null;
        
        // Try to find exact language match
        let voice = this.voices.find(v => v.lang === lang);
        if (voice) return voice;
        
        // Try to find language family match (e.g., 'zh' for 'zh-CN')
        const langFamily = lang.split('-')[0];
        voice = this.voices.find(v => v.lang.startsWith(langFamily));
        if (voice) return voice;
        
        // Fallback to default voice
        return this.voices.find(v => v.default) || this.voices[0];
    }

    speak(text, options = {}) {
        if (!this.isSupported) {
            this.showFallbackMessage(text, options.fallbackMessage);
            return false;
        }

        if (!this.isReady) {
            console.warn('Speech synthesis not ready yet');
            this.showFallbackMessage(text, options.fallbackMessage);
            return false;
        }

        try {
            // Cancel any ongoing speech
            speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(text);
            
            // Set voice
            const voice = this.findBestVoice(options.lang || 'zh-CN');
            if (voice) {
                utterance.voice = voice;
            }
            
            // Set speech parameters
            utterance.lang = options.lang || 'zh-CN';
            utterance.rate = options.rate || 0.8;
            utterance.pitch = options.pitch || 1.0;
            utterance.volume = options.volume || 1.0;

            // Event handlers
            utterance.onstart = () => {
                console.log('Speech started:', text);
                if (options.onStart) options.onStart();
            };

            utterance.onend = () => {
                console.log('Speech ended:', text);
                if (options.onEnd) options.onEnd();
            };

            utterance.onerror = (event) => {
                console.error('Speech error:', event.error);
                this.showFallbackMessage(text, options.fallbackMessage);
                if (options.onError) options.onError(event);
            };

            // Speak the text
            speechSynthesis.speak(utterance);
            return true;

        } catch (error) {
            console.error('Speech synthesis error:', error);
            this.showFallbackMessage(text, options.fallbackMessage);
            return false;
        }
    }

    showFallbackMessage(text, fallbackMessage) {
        const message = fallbackMessage || `Pronunciation: ${text}`;
        
        // Create a temporary tooltip or alert
        if (typeof window !== 'undefined') {
            // Try to show a nice tooltip first
            this.showTooltip(message) || alert(message);
        }
    }

    showTooltip(message) {
        try {
            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.textContent = message;
            tooltip.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
                z-index: 10000;
                pointer-events: none;
                animation: fadeInOut 2s ease-in-out;
            `;

            // Add CSS animation
            if (!document.getElementById('speech-tooltip-style')) {
                const style = document.createElement('style');
                style.id = 'speech-tooltip-style';
                style.textContent = `
                    @keyframes fadeInOut {
                        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                        20% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                        80% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                        100% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                    }
                `;
                document.head.appendChild(style);
            }

            document.body.appendChild(tooltip);

            // Remove tooltip after animation
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.parentNode.removeChild(tooltip);
                }
            }, 2000);

            return true;
        } catch (error) {
            console.error('Tooltip error:', error);
            return false;
        }
    }

    // Convenience method for Chinese names
    speakChineseName(chineseName, fallbackPronunciation) {
        return this.speak(chineseName, {
            lang: 'zh-CN',
            rate: 0.8,
            pitch: 1.0,
            fallbackMessage: fallbackPronunciation || `Pronunciation: ${chineseName}`
        });
    }
}

// Global instance
window.speechUtils = new SpeechUtils();

// Convenience function for easy use in HTML
window.speakName = function(name, pronunciation) {
    return window.speechUtils.speakChineseName(name, pronunciation);
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpeechUtils;
}