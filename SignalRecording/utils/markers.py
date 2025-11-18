import pyqtgraph as pg
class MarkerItem(pg.InfiniteLine):
    """Custom marker line with label and color."""
    
    def __init__(self, pos, label, color='r', movable=True):
        super().__init__(pos=pos, angle=90, movable=movable, pen=pg.mkPen(color, width=2))
        self.label_text = label
        self.marker_color = color
        
        # Create text label
        self.text_label = pg.TextItem(text=label, color=color, anchor=(0, 1))
        self.update_label_position()
        
    def update_label_position(self):
        """Update label position to follow the line."""
        self.text_label.setPos(self.value(), 0)