// Main JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // Color picker functionality
    const colorPicker = document.getElementById('brand_color');
    if (colorPicker) {
        colorPicker.addEventListener('input', function() {
            this.nextElementSibling.textContent = this.value.toUpperCase();
        });
    }

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = this.querySelectorAll('[required]');
            let valid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    valid = false;
                    field.style.borderColor = '#dc3545';
                } else {
                    field.style.borderColor = '';
                }
            });

            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Utility function for API calls
async function apiCall(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        return null;
    }
}

// Color suggestion functionality
function suggestColors() {
    const idea = document.getElementById('campaign_idea').value;
    const brand = document.getElementById('brand_name').value;
    const audience = document.getElementById('target_audience').value;

    if (idea && brand) {
        const suggestColorsBtn = document.getElementById('suggestColors');
        suggestColorsBtn.textContent = '🔄 Generating...';
        suggestColorsBtn.disabled = true;

        fetch('/api/suggest-colors', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                idea: idea,
                brand_name: brand,
                target_audience: audience
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                displayColorSuggestion(data[0]);
            } else {
                // Fallback to default colors
                displayDefaultColors();
            }
            suggestColorsBtn.textContent = '🎨 Suggest New Colors';
            suggestColorsBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            displayDefaultColors();
            suggestColorsBtn.textContent = '🎨 Suggest Colors';
            suggestColorsBtn.disabled = false;
        });
    } else {
        alert('Please enter both campaign idea and brand name first.');
    }
}

function displayColorSuggestion(palette) {
    // Update color boxes
    document.getElementById('primaryColorBox').style.backgroundColor = palette.primary;
    document.getElementById('secondaryColorBox').style.backgroundColor = palette.secondary;
    document.getElementById('accentColorBox').style.backgroundColor = palette.accent;
    
    // Update color values
    document.getElementById('primaryColorValue').textContent = palette.primary;
    document.getElementById('secondaryColorValue').textContent = palette.secondary;
    document.getElementById('accentColorValue').textContent = palette.accent;
    
    // Update reasoning
    if (palette.reasoning) {
        const reasoningText = `${palette.reasoning.primary} ${palette.reasoning.overall}`;
        document.getElementById('colorReasoningText').textContent = reasoningText;
    }
    
    // Update selected color
    document.getElementById('brand_color').value = palette.primary;
    document.getElementById('brandColorValue').textContent = palette.primary;
    
    // Show the section
    document.getElementById('aiSuggestedColors').style.display = 'block';
}

function displayDefaultColors() {
    const defaultColors = {
        primary: '#4A90E2',
        secondary: '#FF6B6B',
        accent: '#4ECDC4',
        reasoning: {
            primary: 'Trustworthy blue evokes professionalism and reliability.',
            overall: 'This balanced palette works well for most brands targeting women and seniors.'
        }
    };
    displayColorSuggestion(defaultColors);
}

// Add click events to color boxes
document.getElementById('primaryColorBox').addEventListener('click', function() {
    const color = document.getElementById('primaryColorValue').textContent;
    setSelectedColor(color);
});

document.getElementById('secondaryColorBox').addEventListener('click', function() {
    const color = document.getElementById('secondaryColorValue').textContent;
    setSelectedColor(color);
});

document.getElementById('accentColorBox').addEventListener('click', function() {
    const color = document.getElementById('accentColorValue').textContent;
    setSelectedColor(color);
});

function setSelectedColor(color) {
    document.getElementById('brand_color').value = color;
    document.getElementById('brandColorValue').textContent = color;
}

// Color picker change event
document.getElementById('brand_color').addEventListener('input', function() {
    document.getElementById('brandColorValue').textContent = this.value;
});

// SIMPLIFIED COLOR SUGGESTION - WORKING VERSION
function suggestColors() {
    const idea = document.getElementById('campaign_idea').value;
    const brand = document.getElementById('brand_name').value;
    const audience = document.getElementById('target_audience').value;

    if (!idea || !brand) {
        alert('Please enter both campaign idea and brand name first.');
        return;
    }

    const suggestColorsBtn = document.getElementById('suggestColors');
    suggestColorsBtn.textContent = '🔄 Generating...';
    suggestColorsBtn.disabled = true;

    console.log('Requesting color suggestions...');

    fetch('/api/suggest-colors', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            idea: idea,
            brand_name: brand,
            target_audience: audience
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Color suggestions received:', data);
        
        if (data && data.length > 0) {
            displayColors(data[0]);
        } else {
            displayDefaultColors();
        }
        
        suggestColorsBtn.textContent = '🎨 Suggest New Colors';
        suggestColorsBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error fetching colors:', error);
        displayDefaultColors();
        suggestColorsBtn.textContent = '🎨 Suggest Colors';
        suggestColorsBtn.disabled = false;
        alert('Using default colors. AI suggestion failed.');
    });
}

function displayColors(palette) {
    console.log('Displaying palette:', palette);
    
    // Set default values if any are missing
    const primary = palette.primary || '#4A90E2';
    const secondary = palette.secondary || '#FF6B6B';
    const accent = palette.accent || '#4ECDC4';
    
    // Update color swatches
    document.getElementById('primarySwatch').style.backgroundColor = primary;
    document.getElementById('secondarySwatch').style.backgroundColor = secondary;
    document.getElementById('accentSwatch').style.backgroundColor = accent;
    
    // Update color values (ALWAYS set text content)
    document.getElementById('primaryValue').textContent = primary;
    document.getElementById('secondaryValue').textContent = secondary;
    document.getElementById('accentValue').textContent = accent;
    
    // Update reasoning
    let reasoningText = 'This color palette is designed for your target audience. ';
    if (palette.reasoning) {
        reasoningText = palette.reasoning.primary + ' ' + (palette.reasoning.overall || '');
    }
    document.getElementById('reasoningText').textContent = reasoningText;
    
    // Update selected color to primary
    document.getElementById('brand_color').value = primary;
    document.getElementById('selectedColorValue').textContent = primary;
    
    // Show the color display area
    document.getElementById('colorDisplay').style.display = 'block';
}

function displayDefaultColors() {
    const defaultPalette = {
        primary: '#4A90E2',
        secondary: '#FF6B6B',
        accent: '#4ECDC4',
        reasoning: {
            primary: 'Trustworthy blue for professionalism, energetic red for calls to action.',
            overall: 'This balanced palette works well for women and senior audiences.'
        }
    };
    displayColors(defaultPalette);
}

// Add click events to color swatches
document.getElementById('primarySwatch').addEventListener('click', function() {
    const color = document.getElementById('primaryValue').textContent;
    document.getElementById('brand_color').value = color;
    document.getElementById('selectedColorValue').textContent = color;
});

document.getElementById('secondarySwatch').addEventListener('click', function() {
    const color = document.getElementById('secondaryValue').textContent;
    document.getElementById('brand_color').value = color;
    document.getElementById('selectedColorValue').textContent = color;
});

document.getElementById('accentSwatch').addEventListener('click', function() {
    const color = document.getElementById('accentValue').textContent;
    document.getElementById('brand_color').value = color;
    document.getElementById('selectedColorValue').textContent = color;
});

// Color picker change event
document.getElementById('brand_color').addEventListener('input', function() {
    document.getElementById('selectedColorValue').textContent = this.value;
});

// Initialize event listener for suggest colors button
document.getElementById('suggestColors').addEventListener('click', suggestColors);