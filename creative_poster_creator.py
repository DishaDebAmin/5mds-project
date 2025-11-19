from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import math

class CreativePosterCreator:
    def __init__(self):
        self.templates = {
            'modern': self._create_modern_poster,
            'creative': self._create_creative_poster,
            'minimal': self._create_minimal_poster,
            'bold': self._create_bold_poster,
            'elegant': self._create_elegant_poster,
            'playful': self._create_playful_poster
        }
        
        self.gradients = {
            'blue_purple': ['#667eea', '#764ba2'],
            'sunset': ['#f093fb', '#f5576c'],
            'ocean': ['#4facfe', '#00f2fe'],
            'forest': ['#43e97b', '#38f9d7'],
            'warm': ['#fa709a', '#fee140'],
            'cool': ['#a8edea', '#fed6e3']
        }
    
    def create_creative_poster(self, brand_name, tagline, ad_copy, logo_path, 
                             color_palette, style='creative', poster_prompt=""):
        """Create a highly creative social media poster"""
        width, height = 1080, 1350  # Instagram portrait format
        poster = Image.new('RGB', (width, height), '#FFFFFF')
        draw = ImageDraw.Draw(poster)
        
        # Choose template based on style and prompt
        template_func = self.templates.get(style, self._create_creative_poster)
        
        # Apply creative background based on prompt
        self._apply_creative_background(draw, width, height, color_palette, poster_prompt)
        
        # Create poster content
        template_func(draw, width, height, brand_name, tagline, ad_copy, color_palette, poster_prompt)
        
        # Add logo if provided
        if logo_path and os.path.exists(logo_path.lstrip('/')):
            try:
                logo = Image.open(logo_path.lstrip('/')).convert('RGBA')
                logo = logo.resize((120, 120))
                poster.paste(logo, (width - 150, 50), logo)
            except Exception as e:
                print(f"Could not add logo: {e}")
        
        # Add final creative touches
        self._add_creative_touches(draw, width, height, color_palette, poster_prompt)
        
        # Save poster
        filename = f"poster_{brand_name.replace(' ', '_').lower()}_{random.randint(1000,9999)}.png"
        filepath = os.path.join('static', 'generated', filename)
        poster.save(filepath, quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'dimensions': f"{width}x{height}",
            'prompt_inspired': poster_prompt[:50] + "..." if len(poster_prompt) > 50 else poster_prompt
        }
    
    def _apply_creative_background(self, draw, width, height, color_palette, prompt):
        """Apply creative background based on prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['nature', 'organic', 'green', 'leaf']):
            self._create_nature_background(draw, width, height, color_palette)
        elif any(word in prompt_lower for word in ['tech', 'digital', 'future', 'innovation']):
            self._create_tech_background(draw, width, height, color_palette)
        elif any(word in prompt_lower for word in ['luxury', 'elegant', 'premium', 'gold']):
            self._create_luxury_background(draw, width, height, color_palette)
        elif any(word in prompt_lower for word in ['fun', 'playful', 'colorful', 'happy']):
            self._create_playful_background(draw, width, height, color_palette)
        else:
            self._create_abstract_background(draw, width, height, color_palette)
    
    def _create_nature_background(self, draw, width, height, colors):
        """Create nature-inspired background"""
        # Gradient background
        for i in range(height):
            ratio = i / height
            r = int(int(colors['primary'][1:3], 16) * (1 - ratio) + int(colors['secondary'][1:3], 16) * ratio)
            g = int(int(colors['primary'][3:5], 16) * (1 - ratio) + int(colors['secondary'][3:5], 16) * ratio)
            b = int(int(colors['primary'][5:7], 16) * (1 - ratio) + int(colors['secondary'][5:7], 16) * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        # Organic shapes
        for i in range(8):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(50, 200)
            self._draw_organic_shape(draw, x, y, size, colors['accent'])
    
    def _create_tech_background(self, draw, width, height, colors):
        """Create tech-inspired background"""
        # Dark background with grid
        draw.rectangle([0, 0, width, height], fill=colors['primary'])
        
        # Grid lines
        for i in range(0, width, 40):
            draw.line([(i, 0), (i, height)], fill=colors['secondary'], width=1)
        for i in range(0, height, 40):
            draw.line([(0, i), (width, i)], fill=colors['secondary'], width=1)
        
        # Glowing nodes
        for i in range(15):
            x = random.randint(50, width-50)
            y = random.randint(50, height-50)
            self._draw_glowing_circle(draw, x, y, random.randint(10, 30), colors['accent'])
    
    def _create_luxury_background(self, draw, width, height, colors):
        """Create luxury-inspired background"""
        # Rich solid color with subtle pattern
        draw.rectangle([0, 0, width, height], fill=colors['primary'])
        
        # Subtle geometric pattern
        for i in range(0, width, 60):
            for j in range(0, height, 60):
                if (i + j) % 120 == 0:
                    draw.rectangle([i, j, i+20, j+20], fill=colors['secondary'], width=0)
        
        # Gold accents
        for i in range(5):
            x = random.randint(0, width)
            y = random.randint(0, height)
            self._draw_decorative_element(draw, x, y, colors['accent'])
    
    def _create_playful_background(self, draw, width, height, colors):
        """Create playful background with random shapes"""
        # Bright background
        draw.rectangle([0, 0, width, height], fill=colors['primary'])
        
        # Random colorful shapes
        for i in range(20):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(30, 100)
            shape_type = random.choice(['circle', 'triangle', 'star', 'heart'])
            color = random.choice([colors['secondary'], colors['accent'], colors['neutral']])
            
            if shape_type == 'circle':
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            elif shape_type == 'triangle':
                self._draw_triangle(draw, x, y, size, color)
            elif shape_type == 'star':
                self._draw_star(draw, x, y, size, color)
            else:  # heart
                self._draw_heart(draw, x, y, size, color)
    
    def _create_abstract_background(self, draw, width, height, colors):
        """Create abstract artistic background"""
        # Fluid gradient
        for i in range(width):
            ratio = i / width
            r = int(int(colors['primary'][1:3], 16) * (1 - ratio) + int(colors['accent'][1:3], 16) * ratio)
            g = int(int(colors['primary'][3:5], 16) * (1 - ratio) + int(colors['accent'][3:5], 16) * ratio)
            b = int(int(colors['primary'][5:7], 16) * (1 - ratio) + int(colors['accent'][5:7], 16) * ratio)
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
        
        # Abstract wave forms
        for i in range(3):
            y_base = height // 4 * (i + 1)
            points = []
            for x in range(0, width + 10, 10):
                y = y_base + math.sin(x * 0.02 + i) * 40
                points.append((x, y))
            
            for j in range(len(points) - 1):
                draw.line([points[j], points[j+1]], fill=colors['secondary'], width=3)
    
    def _create_modern_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Modern poster with clean design"""
        # Content area with subtle background
        content_bg = self._adjust_brightness(colors['primary'], 1.2)
        draw.rectangle([50, 200, width-50, height-200], fill=content_bg, outline=colors['secondary'], width=3)
        
        # Brand name
        self._add_text(draw, width//2, 300, brand_name, colors['secondary'], 72, 'bold')
        
        # Tagline
        self._add_text(draw, width//2, 380, tagline, colors['accent'], 32, 'medium')
        
        # Ad copy
        lines = self._wrap_text(ad_copy['description'], 36, width - 200)
        for i, line in enumerate(lines):
            self._add_text(draw, width//2, 480 + i*45, line, colors['secondary'], 28, 'regular')
        
        # CTA Button
        self._draw_modern_button(draw, width//2, 750, ad_copy['cta'], colors['accent'], colors['primary'])
        
        # Hashtags
        hashtag_text = " ".join(ad_copy['hashtags'])
        self._add_text(draw, width//2, 850, hashtag_text, colors['secondary'], 22, 'light')
    
    def _create_creative_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Creative poster with artistic elements"""
        # Asymmetric layout
        self._add_text(draw, 100, 250, brand_name, colors['primary'], 64, 'bold')
        
        # Creative tagline placement
        self._add_text(draw, width-100, 350, tagline, colors['accent'], 28, 'medium', 'right')
        
        # Creative text blocks
        lines = self._wrap_text(ad_copy['description'], 32, 400)
        for i, line in enumerate(lines):
            y_pos = 450 + i*60
            # Alternate alignment for creative look
            align = 'left' if i % 2 == 0 else 'right'
            x_pos = 100 if align == 'left' else width - 100
            self._add_text(draw, x_pos, y_pos, line, colors['secondary'], 24, 'regular', align)
        
        # Creative CTA
        self._draw_creative_button(draw, width//2, 800, ad_copy['cta'], colors)
        
        # Decorative elements
        self._add_decorative_elements(draw, width, height, colors)
    
    def _create_minimal_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Minimal poster with lots of whitespace"""
        # Very sparse design
        self._add_text(draw, width//2, height//3, brand_name, colors['primary'], 60, 'light')
        
        # Subtle tagline
        self._add_text(draw, width//2, height//3 + 80, tagline, colors['secondary'], 24, 'light')
        
        # Minimal ad copy (first sentence only)
        main_text = ad_copy['description'].split('.')[0] + '.'
        self._add_text(draw, width//2, height//2, main_text, colors['secondary'], 20, 'regular')
        
        # Understated CTA
        self._add_text(draw, width//2, height//2 + 100, ad_copy['cta'], colors['accent'], 28, 'medium')
        
        # Bottom aligned hashtags
        self._add_text(draw, width//2, height - 100, " ".join(ad_copy['hashtags']), colors['secondary'], 18, 'light')
    
    def _create_bold_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Bold poster with strong visual impact"""
        # Full-bleed color blocks
        draw.rectangle([0, 0, width, 300], fill=colors['primary'])
        draw.rectangle([0, height-300, width, height], fill=colors['secondary'])
        
        # Large brand name
        self._add_text(draw, width//2, 150, brand_name, '#FFFFFF', 84, 'bold')
        
        # Contrasting tagline
        self._add_text(draw, width//2, 250, tagline, colors['accent'], 36, 'bold')
        
        # Bold ad copy
        lines = self._wrap_text(ad_copy['description'], 32, width - 100)
        for i, line in enumerate(lines):
            self._add_text(draw, width//2, 400 + i*50, line, colors['primary'], 28, 'bold')
        
        # Prominent CTA
        self._draw_bold_button(draw, width//2, 700, ad_copy['cta'], colors)
    
    def _create_elegant_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Elegant poster with sophisticated typography"""
        # Sophisticated background
        for i in range(height):
            ratio = i / height
            r = int(255 * (1 - ratio) + int(colors['primary'][1:3], 16) * ratio)
            g = int(255 * (1 - ratio) + int(colors['primary'][3:5], 16) * ratio)
            b = int(255 * (1 - ratio) + int(colors['primary'][5:7], 16) * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        # Elegant typography
        self._add_text(draw, width//2, 280, brand_name, colors['secondary'], 68, 'light')
        
        # Decorative line
        draw.line([(width//2-100, 320), (width//2+100, 320)], fill=colors['accent'], width=2)
        
        # Tagline in elegant script style
        self._add_text(draw, width//2, 370, tagline, colors['secondary'], 26, 'regular')
        
        # Refined ad copy
        lines = self._wrap_text(ad_copy['description'], 30, width - 150)
        for i, line in enumerate(lines):
            self._add_text(draw, width//2, 450 + i*42, line, colors['secondary'], 20, 'light')
        
        # Sophisticated CTA
        self._draw_elegant_button(draw, width//2, 750, ad_copy['cta'], colors)
    
    def _create_playful_poster(self, draw, width, height, brand_name, tagline, ad_copy, colors, prompt):
        """Playful poster with fun elements"""
        # Colorful background
        for i in range(width):
            ratio = i / width
            r = int(int(colors['primary'][1:3], 16) * (1 - ratio) + int(colors['accent'][1:3], 16) * ratio)
            g = int(int(colors['primary'][3:5], 16) * (1 - ratio) + int(colors['accent'][3:5], 16) * ratio)
            b = int(int(colors['primary'][5:7], 16) * (1 - ratio) + int(colors['accent'][5:7], 16) * ratio)
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
        
        # Fun brand name with shadow
        self._add_text_with_shadow(draw, width//2, 250, brand_name, colors['secondary'], colors['primary'], 72, 'bold')
        
        # Playful tagline
        self._add_text(draw, width//2, 350, tagline, colors['accent'], 32, 'medium')
        
        # Fun ad copy with emojis
        description = "✨ " + ad_copy['description'] + " 🌟"
        lines = self._wrap_text(description, 32, width - 100)
        for i, line in enumerate(lines):
            self._add_text(draw, width//2, 450 + i*50, line, colors['secondary'], 26, 'regular')
        
        # Playful CTA
        self._draw_playful_button(draw, width//2, 750, "🎯 " + ad_copy['cta'], colors)
        
        # Fun elements
        self._add_playful_elements(draw, width, height, colors)
    
    def _add_text(self, draw, x, y, text, color, size, weight='regular', align='center'):
        """Add text with specified styling"""
        try:
            # Try different font weights
            font_paths = {
                'light': "arial.ttf",
                'regular': "arial.ttf", 
                'medium': "arialbd.ttf",
                'bold': "arialbd.ttf"
            }
            font = ImageFont.truetype(font_paths.get(weight, "arial.ttf"), size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if align == 'center':
            x_pos = x - text_width // 2
        elif align == 'right':
            x_pos = x - text_width
        else:  # left
            x_pos = x
        
        draw.text((x_pos, y - text_height // 2), text, fill=color, font=font)
    
    def _add_text_with_shadow(self, draw, x, y, text, color, shadow_color, size, weight):
        """Add text with shadow effect"""
        self._add_text(draw, x+2, y+2, text, shadow_color, size, weight)
        self._add_text(draw, x, y, text, color, size, weight)
    
    def _wrap_text(self, text, max_width_chars, line_width):
        """Wrap text to fit within line width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_width_chars:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_modern_button(self, draw, x, y, text, bg_color, text_color):
        """Draw modern button"""
        button_width = 300
        button_height = 70
        draw.rectangle([x-button_width//2, y-button_height//2, x+button_width//2, y+button_height//2], 
                      fill=bg_color, outline=text_color, width=2)
        self._add_text(draw, x, y, text, text_color, 24, 'medium')
    
    def _draw_creative_button(self, draw, x, y, text, colors):
        """Draw creative button"""
        # Irregular shaped button
        points = [
            (x-120, y-25), (x-80, y-35), (x+80, y-35), (x+120, y-25),
            (x+120, y+25), (x+80, y+35), (x-80, y+35), (x-120, y+25)
        ]
        draw.polygon(points, fill=colors['accent'])
        self._add_text(draw, x, y, text, colors['primary'], 22, 'bold')
    
    def _draw_bold_button(self, draw, x, y, text, colors):
        """Draw bold button"""
        button_width = 400
        button_height = 80
        draw.rectangle([x-button_width//2, y-button_height//2, x+button_width//2, y+button_height//2], 
                      fill=colors['accent'])
        self._add_text(draw, x, y, text, colors['primary'], 28, 'bold')
    
    def _draw_elegant_button(self, draw, x, y, text, colors):
        """Draw elegant button"""
        button_width = 250
        button_height = 50
        draw.rectangle([x-button_width//2, y-button_height//2, x+button_width//2, y+button_height//2], 
                      fill=colors['accent'], outline=colors['secondary'], width=1)
        self._add_text(draw, x, y, text, colors['secondary'], 20, 'light')
    
    def _draw_playful_button(self, draw, x, y, text, colors):
        """Draw playful button"""
        # Rounded button with shadow
        button_width = 280
        button_height = 60
        # Shadow
        draw.rectangle([x-button_width//2+4, y-button_height//2+4, x+button_width//2+4, y+button_height//2+4], 
                      fill=colors['secondary'])
        # Button
        draw.rectangle([x-button_width//2, y-button_height//2, x+button_width//2, y+button_height//2], 
                      fill=colors['accent'])
        self._add_text(draw, x, y, text, colors['primary'], 24, 'bold')
    
    def _draw_organic_shape(self, draw, x, y, size, color):
        """Draw organic leaf-like shape"""
        points = []
        for i in range(8):
            angle = i * math.pi / 4
            radius = size * (0.8 + 0.4 * random.random())
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)
    
    def _draw_glowing_circle(self, draw, x, y, radius, color):
        """Draw glowing circle effect"""
        # Main circle
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        # Glow effect
        for i in range(1, 4):
            glow_radius = radius + i * 5
            alpha = 100 - i * 25
            glow_color = color + format(alpha, '02x')
            draw.ellipse([x-glow_radius, y-glow_radius, x+glow_radius, y+glow_radius], 
                        fill=glow_color)
    
    def _draw_decorative_element(self, draw, x, y, color):
        """Draw decorative element for luxury backgrounds"""
        # Simple geometric decorative element
        size = random.randint(20, 40)
        shape = random.choice(['circle', 'diamond', 'star'])
        
        if shape == 'circle':
            draw.ellipse([x-size, y-size, x+size, y+size], outline=color, width=2)
        elif shape == 'diamond':
            points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
            draw.polygon(points, outline=color, width=2)
        else:  # star
            self._draw_star(draw, x, y, size, color, False)
    
    def _draw_triangle(self, draw, x, y, size, color):
        """Draw triangle"""
        points = [
            (x, y - size),
            (x - size * 0.866, y + size/2),
            (x + size * 0.866, y + size/2)
        ]
        draw.polygon(points, fill=color)
    
    def _draw_star(self, draw, x, y, size, color, fill=True):
        """Draw star"""
        points = []
        for i in range(10):
            angle = math.pi/2 + i * math.pi/5
            radius = size if i % 2 == 0 else size/2
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        
        if fill:
            draw.polygon(points, fill=color)
        else:
            draw.polygon(points, outline=color, width=2)
    
    def _draw_heart(self, draw, x, y, size, color):
        """Draw heart"""
        # Simplified heart using two circles and a triangle
        draw.ellipse([x-size, y-size, x, y], fill=color)
        draw.ellipse([x, y-size, x+size, y], fill=color)
        draw.polygon([(x-size, y), (x+size, y), (x, y+size)], fill=color)
    
    def _add_decorative_elements(self, draw, width, height, colors):
        """Add decorative elements to creative poster"""
        for i in range(5):
            x = random.randint(50, width-50)
            y = random.randint(50, height-50)
            self._draw_abstract_shape(draw, x, y, colors['accent'])
    
    def _add_playful_elements(self, draw, width, height, colors):
        """Add playful elements to poster"""
        # Floating shapes
        for i in range(8):
            x = random.randint(50, width-50)
            y = random.randint(50, height-50)
            size = random.randint(10, 30)
            shape = random.choice(['circle', 'star', 'heart'])
            color = random.choice([colors['secondary'], colors['accent']])
            
            if shape == 'circle':
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            elif shape == 'star':
                self._draw_star(draw, x, y, size, color)
            else:  # heart
                self._draw_heart(draw, x, y, size, color)
    
    def _draw_abstract_shape(self, draw, x, y, color):
        """Draw abstract shape"""
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            radius = random.randint(20, 40)
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)
    
    def _add_creative_touches(self, draw, width, height, colors, prompt):
        """Add final creative touches based on prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['sparkle', 'glitter', 'shine']):
            self._add_sparkle_effect(draw, width, height, colors['accent'])
        elif any(word in prompt_lower for word in ['texture', 'grain', 'vintage']):
            self._add_texture_effect(draw, width, height)
        elif any(word in prompt_lower for word in ['glow', 'neon', 'light']):
            self._add_glow_effect(draw, width, height, colors['accent'])
    
    def _add_sparkle_effect(self, draw, width, height, color):
        """Add sparkle effect"""
        for i in range(20):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(2, 6)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    def _add_texture_effect(self, draw, width, height):
        """Add texture/noise effect"""
        for i in range(1000):
            x = random.randint(0, width)
            y = random.randint(0, height)
            brightness = random.randint(240, 255)
            draw.point((x, y), fill=(brightness, brightness, brightness, 50))
    
    def _add_glow_effect(self, draw, width, height, color):
        """Add glow effect around edges"""
        glow_width = 10
        for i in range(glow_width):
            alpha = 100 - i * 10
            glow_color = color + format(alpha, '02x')
            # Top
            draw.line([(i, i), (width-i, i)], fill=glow_color, width=1)
            # Bottom
            draw.line([(i, height-i), (width-i, height-i)], fill=glow_color, width=1)
            # Left
            draw.line([(i, i), (i, height-i)], fill=glow_color, width=1)
            # Right
            draw.line([(width-i, i), (width-i, height-i)], fill=glow_color, width=1)
    
    def _adjust_brightness(self, hex_color, factor):
        """Adjust color brightness"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)