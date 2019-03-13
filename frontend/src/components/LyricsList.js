import React from 'react';
import axios from "axios";
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import IconButton from '@material-ui/core/IconButton';
import DeleteIcon from '@material-ui/icons/Delete';
import EditIcon from '@material-ui/icons/Edit';
import MusicNoteIcon from '@material-ui/icons/MusicNote';
import Avatar from '@material-ui/core/Avatar';
import ListItemAvatar from '@material-ui/core/ListItemAvatar';
import ListItemText from '@material-ui/core/ListItemText';
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction';
import {Link} from 'react-router-dom'

export default class LyricsList extends React.Component {

	state = {
		lyrics: []
	};

	componentDidMount() {
		this.getLyrics();
	}

	getLyrics = () => {
		axios.get("/api/lyrics/").then(result => this.setState({lyrics: result.data}))
	};

	deleteLyric = (lyric_id) => {
		axios.delete(`/api/lyrics/${lyric_id}/`).then(this.getLyrics)

	};



	renderLyrics() {
		const lyrics = Object.values((this.state.lyrics));
		console.log(lyrics);
		return lyrics.map(lyric =>
			<ListItem button component="a" href={`/lyrics/${lyric.id}`}>
				<ListItemAvatar>
					<Avatar style={{backgroundColor: '#006064'}}>
						<MusicNoteIcon/>
					</Avatar>
				</ListItemAvatar>
				<ListItemText primary={lyric.title} secondary={`Last Updated At: ${lyric.updated_at}`}/>
				<ListItemSecondaryAction>
					<IconButton aria-label="Edit" component={Link} to={{pathname: '/lyric', state:{lyric: lyric}}}>
						<EditIcon/>
					</IconButton>
					<IconButton aria-label="Delete" onClick={() => {
						this.deleteLyric(lyric.id)
					}}>
						<DeleteIcon/>
					</IconButton>
				</ListItemSecondaryAction>
			</ListItem>);
	}

	render() {
		return (
			<div>
				<List>
					{this.renderLyrics()}
				</List>
			</div>
		)
	}

}