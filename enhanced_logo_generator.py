from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import math

class EnhancedLogoGenerator:
    def __init__(self):
        self.logo_styles = {
            'modern': self._create_modern_logo,
            'creative': self._create_creative_logo,
            'minimal': self._create_minimal_logo,
            'luxury': self._create_luxury_logo,
            'playful': self._create_playful_logo
        }
        
        self.icon_sets = {
            'modern': ['●', '■', '▲', '◆', '○', '□', '△', '◇'],
            'creative': ['★', '☆', '✶', '✦', '✧', '❂', '✼', '✻'],
            'minimal': ['•', '·', '∘', '○', '□', '△', '▷', '◁'],
            'luxury': ['♠', '♥', '♣', '♦', '♤', '♡', '♧', '♢'],
            'playful': ['☀', '★', '❤', '♦', '♠', '♣', '☁', '☂']
        }
    
    def generate_creative_logo(self, brand_name, primary_color, logo_prompt, style='creative'):
        """Generate creative logo based on user prompt"""
        width, height = 400, 400
        logo = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(logo)
        
        # Choose style-based function
        logo_func = self.logo_styles.get(style, self._create_creative_logo)
        
        # Generate logo based on prompt
        logo_data = logo_func(draw, width, height, brand_name, primary_color, logo_prompt)
        
        # Save logo
        filename = f"logo_{brand_name.replace(' ', '_').lower()}_{random.randint(1000,9999)}.png"
        filepath = os.path.join('static', 'generated', filename)
        logo.save(filepath, quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'primary_color': primary_color,
            'prompt': logo_prompt,
            'description': logo_data.get('description', f'{style} logo for {brand_name}')
        }
    
    def _create_modern_logo(self, draw, width, height, brand_name, primary_color, prompt):
        """Create modern geometric logo"""
        # Background shape
        shape_type = random.choice(['circle', 'squares', 'lines'])
        
        if shape_type == 'circle':
            # Concentric circles
            for i in range(3):
                radius = 150 - i * 40
                color = self._adjust_brightness(primary_color, 1 - i * 0.2)
                draw.ellipse([width//2-radius, height//2-radius, 
                             width//2+radius, height//2+radius], 
                            outline=color, width=8)
        
        elif shape_type == 'squares':
            # Rotated squares
            for i in range(3):
                size = 200 - i * 50
                angle = i * 15
                self._draw_rotated_square(draw, width//2, height//2, size, angle, primary_color)
        
        # Icon and text
        icon = random.choice(self.icon_sets['modern'])
        self._add_logo_text(draw, width, height, brand_name, primary_color)
        self._add_centered_icon(draw, width, height, icon, primary_color, 80)
        
        return {'description': f'Modern geometric logo inspired by {prompt}'}
    
    def _create_creative_logo(self, draw, width, height, brand_name, primary_color, prompt):
        """Create creative abstract logo"""
        # Abstract background
        for i in range(8):
            angle = i * math.pi / 4
            x1 = width//2 + math.cos(angle) * 50
            y1 = height//2 + math.sin(angle) * 50
            x2 = width//2 + math.cos(angle) * 180
            y2 = height//2 + math.sin(angle) * 180
            
            draw.line([(x1, y1), (x2, y2)], fill=primary_color, width=6)
        
        # Central element based on prompt
        if any(word in prompt.lower() for word in ['nature', 'leaf', 'tree', 'plant']):
            self._draw_nature_element(draw, width, height, primary_color)
        elif any(word in prompt.lower() for word in ['tech', 'digital', 'future', 'innovation']):
            self._draw_tech_element(draw, width, height, primary_color)
        elif any(word in prompt.lower() for word in ['food', 'restaurant', 'cafe', 'kitchen']):
            self._draw_food_element(draw, width, height, primary_color)
        else:
            self._draw_abstract_element(draw, width, height, primary_color)
        
        self._add_logo_text(draw, width, height, brand_name, primary_color)
        
        return {'description': f'Creative abstract logo inspired by "{prompt}"'}
    
    def _create_minimal_logo(self, draw, width, height, brand_name, primary_color, prompt):
        """Create minimal logo"""
        # Simple geometric shape
        shape = random.choice(['circle', 'square', 'triangle'])
        size = 120
        
        if shape == 'circle':
            draw.ellipse([width//2-size, height//2-size, width//2+size, height//2+size], 
                        outline=primary_color, width=4)
        elif shape == 'square':
            draw.rectangle([width//2-size, height//2-size, width//2+size, height//2+size], 
                          outline=primary_color, width=4)
        else:  # triangle
            self._draw_triangle(draw, width//2, height//2, size, primary_color)
        
        # Simple icon
        icon = random.choice(self.icon_sets['minimal'])
        self._add_centered_icon(draw, width, height, icon, primary_color, 40)
        self._add_minimal_text(draw, width, height, brand_name, primary_color)
        
        return {'description': f'Minimal logo reflecting {prompt}'}
    
    def _create_luxury_logo(self, draw, width, height, brand_name, primary_color, prompt):
        """Create luxury emblem logo"""
        # Circular emblem
        draw.ellipse([width//2-160, height//2-160, width//2+160, height//2+160], 
                    outline=primary_color, width=6)
        draw.ellipse([width//2-120, height//2-120, width//2+120, height//2+120], 
                    outline=primary_color, width=3)
        
        # Decorative elements
        for i in range(8):
            angle = i * math.pi / 4
            x = width//2 + math.cos(angle) * 140
            y = height//2 + math.sin(angle) * 140
            draw.ellipse([x-10, y-10, x+10, y+10], fill=primary_color)
        
        # Central icon
        icon = random.choice(self.icon_sets['luxury'])
        self._add_centered_icon(draw, width, height, icon, '#FFFFFF', 60)
        
        # Brand name in circular text (simplified)
        self._add_emblem_text(draw, width, height, brand_name, primary_color)
        
        return {'description': f'Luxury emblem logo for {brand_name}'}
    
    def _create_playful_logo(self, draw, width, height, brand_name, primary_color, prompt):
        """Create playful, fun logo"""
        # Colorful background elements
        colors = [primary_color, self._adjust_brightness(primary_color, 1.3),
                 self._adjust_brightness(primary_color, 0.7)]
        
        for i in range(5):
            x = random.randint(50, width-50)
            y = random.randint(50, height-50)
            size = random.randint(30, 80)
            color = random.choice(colors)
            
            shape_type = random.choice(['circle', 'star', 'heart'])
            if shape_type == 'circle':
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            elif shape_type == 'star':
                self._draw_star(draw, x, y, size, color)
            else:  # heart
                self._draw_heart(draw, x, y, size, color)
        
        # Central icon
        icon = random.choice(self.icon_sets['playful'])
        self._add_centered_icon(draw, width, height, icon, '#FFFFFF', 60)
        self._add_playful_text(draw, width, height, brand_name, primary_color)
        
        return {'description': f'Playful logo inspired by {prompt}'}
    
    def _draw_nature_element(self, draw, width, height, color):
        """Draw nature-inspired element"""
        # Simple leaf/tree design
        for i in range(5):
            angle = i * 2 * math.pi / 5
            x1 = width//2
            y1 = height//2
            x2 = width//2 + math.cos(angle) * 80
            y2 = height//2 + math.sin(angle) * 80
            
            draw.line([(x1, y1), (x2, y2)], fill=color, width=8)
            draw.ellipse([x2-15, y2-15, x2+15, y2+15], fill=color)
    
    def _draw_tech_element(self, draw, width, height, color):
        """Draw tech-inspired element"""
        # Circuit board pattern
        points = []
        for i in range(6):
            angle = i * 2 * math.pi / 6
            x = width//2 + math.cos(angle) * 60
            y = height//2 + math.sin(angle) * 60
            points.append((x, y))
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                draw.line([points[i], points[j]], fill=color, width=3)
    
    def _draw_food_element(self, draw, width, height, color):
        """Draw food-inspired element"""
        # Simple utensil or food item
        draw.ellipse([width//2-40, height//2-40, width//2+40, height//2+40], 
                    outline=color, width=4)
        draw.line([(width//2, height//2-40), (width//2, height//2+40)], 
                 fill=color, width=4)
        draw.line([(width//2-40, height//2), (width//2+40, height//2)], 
                 fill=color, width=4)
    
    def _draw_abstract_element(self, draw, width, height, color):
        """Draw abstract artistic element"""
        # Flowing curves
        for i in range(3):
            start_angle = random.uniform(0, 2 * math.pi)
            end_angle = start_angle + random.uniform(0.5, 1.5)
            radius = random.randint(60, 120)
            
            self._draw_arc(draw, width//2, height//2, radius, start_angle, end_angle, color, 6)
    
    def _draw_rotated_square(self, draw, cx, cy, size, angle, color):
        """Draw rotated square"""
        points = []
        for i in range(4):
            point_angle = angle + i * math.pi / 2
            x = cx + math.cos(point_angle) * size / math.sqrt(2)
            y = cy + math.sin(point_angle) * size / math.sqrt(2)
            points.append((x, y))
        
        draw.polygon(points, outline=color, width=4)
    
    def _draw_triangle(self, draw, cx, cy, size, color):
        """Draw equilateral triangle"""
        points = [
            (cx, cy - size),
            (cx - size * math.sqrt(3)/2, cy + size/2),
            (cx + size * math.sqrt(3)/2, cy + size/2)
        ]
        draw.polygon(points, outline=color, width=4)
    
    def _draw_star(self, draw, cx, cy, size, color):
        """Draw star shape"""
        points = []
        for i in range(10):
            angle = math.pi/2 + i * math.pi/5
            radius = size if i % 2 == 0 else size/2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)
    
    def _draw_heart(self, draw, cx, cy, size, color):
        """Draw heart shape"""
        # Simplified heart using two circles and a triangle
        draw.ellipse([cx-size, cy-size, cx, cy], fill=color)
        draw.ellipse([cx, cy-size, cx+size, cy], fill=color)
        draw.polygon([(cx-size, cy), (cx+size, cy), (cx, cy+size)], fill=color)
    
    def _draw_arc(self, draw, cx, cy, radius, start_angle, end_angle, color, width):
        """Draw an arc"""
        steps = 20
        points = []
        for i in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * i / steps
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=width)
    
    def _add_logo_text(self, draw, width, height, brand_name, color):
        """Add brand name text to logo"""
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), brand_name, font=font)
        text_width = bbox[2] - bbox[0]
        
        draw.text((width//2 - text_width//2, height - 60), 
                 brand_name, fill=color, font=font)
    
    def _add_minimal_text(self, draw, width, height, brand_name, color):
        """Add minimal text"""
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Use initials for minimal logo
        initials = ''.join([word[0].upper() for word in brand_name.split()])
        if len(initials) > 3:
            initials = initials[:3]
        
        bbox = draw.textbbox((0, 0), initials, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text((width//2 - text_width//2, height//2 - text_height//2), 
                 initials, fill=color, font=font)
    
    def _add_emblem_text(self, draw, width, height, brand_name, color):
        """Add text around emblem"""
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        # Simplified version - just place text below
        bbox = draw.textbbox((0, 0), brand_name, font=font)
        text_width = bbox[2] - bbox[0]
        
        draw.text((width//2 - text_width//2, height - 50), 
                 brand_name, fill=color, font=font)
    
    def _add_playful_text(self, draw, width, height, brand_name, color):
        """Add playful text"""
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), brand_name, font=font)
        text_width = bbox[2] - bbox[0]
        
        draw.text((width//2 - text_width//2, height - 70), 
                 brand_name, fill=color, font=font)
    
    def _add_centered_icon(self, draw, width, height, icon, color, size):
        """Add centered icon"""
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), icon, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text((width//2 - text_width//2, height//2 - text_height//2), 
                 icon, fill=color, font=font)
    
    def _adjust_brightness(self, hex_color, factor):
        """Adjust color brightness"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def generate_tagline(self, brand_name, campaign_idea, industry, logo_prompt):
        """Generate creative tagline based on logo prompt"""
        tagline_themes = {
            'modern': [
                f"Redefining Excellence with {brand_name}",
                f"Where Innovation Meets Style - {brand_name}",
                f"Modern Solutions for Modern Times",
                f"The Future is {brand_name}"
            ],
            'creative': [
                f"Unleash Your Creativity with {brand_name}",
                f"Where Ideas Come to Life",
                f"Innovation Through Imagination - {brand_name}",
                f"Create the Extraordinary"
            ],
            'minimal': [
                f"Simplicity Perfected by {brand_name}",
                f"Less is More, Perfected",
                f"Essential Excellence - {brand_name}",
                f"Pure and Simple"
            ],
            'luxury': [
                f"Experience Unparalleled Luxury with {brand_name}",
                f"Where Excellence Becomes Tradition",
                f"The Epitome of Elegance - {brand_name}",
                f"Luxury Redefined"
            ],
            'playful': [
                f"Fun Meets Function with {brand_name}",
                f"Where Joy Meets Innovation",
                f"Playful Excellence - {brand_name}",
                f"Making Life More Fun"
            ]
        }
        
        # Use prompt words to customize tagline
        prompt_words = logo_prompt.lower()
        custom_taglines = []
        
        if any(word in prompt_words for word in ['nature', 'eco', 'green']):
            custom_taglines.extend([
                f"Naturally Better with {brand_name}",
                f"Eco-Friendly Innovation",
                f"Green Living, Modern Solutions"
            ])
        
        if any(word in prompt_words for word in ['tech', 'digital', 'future']):
            custom_taglines.extend([
                f"Tomorrow's Technology Today",
                f"Digital Innovation, Real Results",
                f"The Future in Your Hands"
            ])
        
        if any(word in prompt_words for word in ['food', 'taste', 'fresh']):
            custom_taglines.extend([
                f"Taste the Difference with {brand_name}",
                f"Fresh Ideas, Better Taste",
                f"Culinary Excellence Redefined"
            ])
        
        # Combine theme-based and custom taglines
        theme_taglines = tagline_themes.get('creative', tagline_themes['modern'])
        all_taglines = custom_taglines + theme_taglines
        
        return random.choice(all_taglines) if all_taglines else f"Excellence by {brand_name}"